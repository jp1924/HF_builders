import base64
import io
import json
import multiprocessing as mp
import os
import pickle
import platform
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tarfile import TarFile
from time import sleep, time
from typing import List, Optional, Tuple
from urllib.parse import quote
from zipfile import ZipFile

import datasets
import numpy as np
import pandas as pd
import psutil
import pytz
import requests
import win32clipboard as clipboard
from bs4 import BeautifulSoup
from datasets import (
    BuilderConfig,
    Dataset,
    DatasetInfo,
    Features,
    GeneratorBasedBuilder,
    Image,
    Split,
    SplitGenerator,
    Value,
)
from natsort import natsorted
from openai import OpenAI
from PIL import Image as PIL_Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager


if platform.system() != "Windows":
    raise RuntimeError("이 코드는 무조건 아레 한글이 설치된 windows에서만 동작이 가능합니다.")


from win32com.shell.shell import IsUserAnAdmin  # type: ignore


if not IsUserAnAdmin():
    # NOTE: 왜 이려냐고? appdata/temp 파일의 용량이 높아지면 실행 시간이 늘어나서 그럼, 그래서 appdata 내의 파알에 접근해야 하기 때문에 관한 필요함.
    raise PermissionError("이 코드는 관리자권한인 상태에서만 실행이 가능합니다.")

only_ko_en_chr = re.compile("[^가-힣A-Za-z0-9]")

# NOTE: pickle, maximum recursion depth exceeded error 때문에 선언함.
sys.setrecursionlimit(10000)
timezone = pytz.timezone("Asia/Seoul")

temp_dir_path = tempfile.gettempdir()

get_time = lambda: datetime.now(timezone).strftime("%H:%M:%S")  # noqa: E731

client = OpenAI()


logger = datasets.logging.get_logger()


@dataclass
class Table:
    html: str
    row: int
    col: int


GPT_INSTRUCTION = "Your goal is to generate an ANSWER identical to the EXAMPLE_ANSWER using the content of the given TABLE as a reference. Ensure to generate the ANSWER only for the given QUESTION, and do not respond to the QUESTION with another question. When generating the ANSWER, make sure to refer to the given TABLE. Do not give short answers, and provide a rich and detailed explanation. The generated questions must be written in Korean. It is very important to follow these guidelines and under no circumstances should you deviate from them."


_LICENSE = None
_CITATION = None

_DESCRIPTION = (
    """테이블 이미지가 포함된 일반 문서 내에서 표 내의 특정 값을 탐색하기 위한 기계학습용 질의어와 정답 세트 데이터"""
)


DATASET_KEY = "71565"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/{DATASET_KEY}.do"
_HOMEPAGE = (
    f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"
)

_VERSION = "1.0.0"
_DATANAME = "DocumentTableVQAInst"
DATASET_SIZE = 0.21531  # aihub 데이터의 크기임. GB 단위


def download_file_from_site(dataset: Dataset, base_path: Path) -> None:
    def _set_download_directory(driver: webdriver.Chrome, download_path: str) -> None:
        params = {"behavior": "allow", "downloadPath": download_path}
        driver.execute_cdp_cmd("Page.setDownloadBehavior", params)

    def _open_chrome_windows(max_retries: int = 10) -> webdriver.Chrome:
        service = Service(executable_path=ChromeDriverManager().install())
        for retries in range(max_retries):
            try:
                driver = webdriver.Chrome(service=service)
                return driver
            except BaseException as e:
                _close_chrome_windows()
                logger.debug(f"driver {retries}회 실행 때 다음의 예외가 발생함: {e}")

        exit("chrome driver가 생성되질 않아 종료 함.")

    def _close_chrome_windows() -> None:
        for proc in psutil.process_iter(["pid", "name"]):
            if "chrome" not in proc._name:
                continue
            try:
                proc.terminate()
            except BaseException as e:  # noqa: F841
                pass

    def _download_hwp_file(
        driver: "webdriver",
        doc_title: str,
        doc_published: str,
        download_path: Path,
    ) -> None:
        bid_notice_ls = driver.find_elements(By.CSS_SELECTOR, "ul.search_list > li")
        for bid_notice in bid_notice_ls:
            keyword_ls = bid_notice.find_elements(By.CSS_SELECTOR, "span.keyword")
            bid_notice_title = " ".join([keyword.text for keyword in keyword_ls])

            published_date = bid_notice.find_element(By.CSS_SELECTOR, "ul.info2 > li.m2 > span").text
            bid_notice_published_date = published_date.split(" ")[0].replace("/", "")

            title_flag = bid_notice_title != doc_title
            published_flag = bid_notice_published_date != doc_published
            if title_flag or published_flag:
                continue

            # 해당 리스트 선택
            bid_notice.find_element(By.TAG_NAME, "a").click()
            iframe_src = driver.find_element(By.CSS_SELECTOR, "iframe").get_attribute("src")
            driver.get(iframe_src)
            [tag.find_element(By.CSS_SELECTOR, "a.btn_mdl").click() for tag in driver.find_elements(By.ID, "epDialog")]

            hwp_download_ls = [
                elem.find_elements(By.TAG_NAME, "a")
                for elem in driver.find_elements(By.CSS_SELECTOR, "table.table_list_attchFileTbl")
            ]

            if not hwp_download_ls:
                continue

            hwp_download_ls = [link_tag for link_tag in hwp_download_ls[0] if "hwp" in link_tag.text]

            # NOTE: 이미 다운이 된 hwp 파일이면 해당 사이트는 패스 함.
            if len(hwp_download_ls) == len(list(download_path.glob("*.hwp"))):
                break

            max_retries = 3
            for attempt in range(1, max_retries):
                downloaded_hwp_ls = list()
                for herf_link in hwp_download_ls:
                    herf_link.click()
                    downloaded_hwp_ls.append(herf_link.text)
                    sleep(0.2)

                # NOTE: 혹시 몰라 예상 했던 개수 만큼 초과해서 다운 될 수 있잖어
                if len(list(download_path.glob("*.hwp"))) >= len(downloaded_hwp_ls):
                    break

            driver.back()
            driver.back()

    _close_chrome_windows()
    driver = _open_chrome_windows()

    loading_interval = 0.5
    for data in tqdm(dataset, desc="_download_hwp_files"):
        doc_title = data["metadata"]["doc_title"]
        doc_source = data["metadata"]["doc_source"]
        doc_published = str(data["metadata"]["doc_published"])

        save_path = base_path.joinpath(doc_title.replace("/", "|"))

        if doc_source != "나라장터" or save_path.exists():
            # 중복된 테이블 필터링 수행함.
            continue

        enc_title = quote(doc_title, encoding="euc-kr")
        koneps_url = f"https://www.g2b.go.kr:8340/search.do?category=TGONG&kwd={enc_title}"
        driver.get(koneps_url)

        _set_download_directory(driver, save_path.as_posix())

        sleep(loading_interval)

        try:
            _download_hwp_file(
                driver,
                doc_title,
                doc_published,
                save_path,
            )
        except BaseException as e:
            _close_chrome_windows()
            driver = _open_chrome_windows()
            logger.debug(f"{e} 애러가 발생해 해당 페이지({koneps_url})는 건너 뜀.")

            continue

        if not list(save_path.glob("*.hwp")):
            continue

    driver.close()

    logger.info("hwp file download finish!!!!")


def extract_table_from_file(dataset: Dataset, base_path: Path) -> None:
    from pyhwpx import Hwp

    def _extract_html_table(hwp_dir_path: Path) -> Tuple[List[Table], float]:
        def get_row_col_num(hwp: Hwp) -> Tuple[int, int]:
            # get_row_num, get_col_num의 기능은 겹침. 단순 RowCount냐, ColCount의 차이임. 속도 개선을 위해 이 부분을 간소화 함.
            cur_pos = hwp.get_pos()
            hwp.SelectCtrlFront()
            t = hwp.GetTextFile("HWPML2X", "saveblock")
            root = ET.fromstring(t)
            table = root.find(".//TABLE")
            row_count = int(table.get("RowCount"))
            col_count = int(table.get("ColCount"))
            hwp.set_pos(*cur_pos)
            return (row_count, col_count)

        table_ls = list()
        time_ls = list()
        for hwp_file_path in hwp_dir_path.glob("*.hwp"):
            hwp.open(hwp_file_path.as_posix())

            extract_time_ls = list()
            one_file_table_ls = list()

            ctrl = hwp.HeadCtrl
            while ctrl:
                if ctrl.UserDesc != "표":
                    ctrl = ctrl.Next
                    continue

                start_time = time()

                hwp.SetPosBySet(ctrl.GetAnchorPos(0))
                hwp.HAction.Run("SelectCtrlFront")
                hwp.HAction.Run("Copy")

                try:
                    html = _get_html_from_clipboard()
                    hwp.ShapeObjTableSelCell()
                    table_df = pd.read_html(io.StringIO(html))[0]
                    row_num, col_num = table_df.shape

                    # NOTE: 나중에 알게 되었는데, 이거 부정확 함.
                    #       row 10개를 1개로 판단하거나 그럼, 이건 그냥 df로 해서 확인하는게 가장 정확할 듯.
                    # row_num, col_num = get_row_col_num(hwp)

                    # read_html으로 파싱 되는지 확인 한번 함.
                except BaseException as e:
                    logger.debug(f"table 추출 중 다음과 같은 애러가 발생: {e}")
                    ctrl = ctrl.Next
                    continue

                if not row_num or not col_num:
                    ctrl = ctrl.Next
                    continue

                table = Table(html=html, col=col_num, row=row_num)

                one_file_table_ls.append(table)
                ctrl = ctrl.Next

                end_time = time()

                extract_time_ls.append(end_time - start_time)
            table_ls.extend(one_file_table_ls)

            pickle_save_path = hwp_file_path.parent.joinpath(f"{hwp_file_path.stem}.pickle")
            pickle_save_path.write_bytes(pickle.dumps(one_file_table_ls))
            time_ls.extend(extract_time_ls)

        mean_time = sum(time_ls) / len(time_ls)
        return (table_ls, mean_time)

    def _get_html_from_clipboard(max_retries: int = 10) -> Optional[str]:
        for attempt in range(1, max_retries):
            try:
                clipboard.OpenClipboard()
                html_format = clipboard.RegisterClipboardFormat("HTML Format")
                html = clipboard.GetClipboardData(html_format)
                clipboard.EmptyClipboard()
                clipboard.CloseClipboard()

                html = html.decode("utf-8", errors="ignore")
                html_soup = BeautifulSoup(html, "html.parser")
                html_table = html_soup.find("html").find("table")
                html_table = html_table.encode().decode("utf-8")

                return html_table
            except Exception as e:
                if attempt < max_retries:
                    sleep(0.1)
                else:
                    logger.debug(f"Failed to access clipboard after {max_retries} attempts")
                    raise e
            finally:
                try:
                    # 종종 clipboard가 닫히지 않는 경우가 있음.
                    clipboard.CloseClipboard()
                except BaseException as e:  # noqa: F841
                    pass

        return None

    def _open_hwp_windows(max_retries: int = 10) -> Hwp:
        for retries in range(max_retries):
            try:
                # NOTE: visible False로 하면 추출 속도가 더 빨라지기는 하는데
                #       종종 hwp에서 코드로는 처리 못하는 경고창이나 알수없는 무한루프가 걸리는 경우가 있음.
                #       그거 확인하기 위해선 visible이 필요함.
                hwp = Hwp(visible=True)
                return hwp
            except BaseException as e:
                _close_hwp_windows()
                logger.debug(f"hwp {retries}회 실행 때 다음의 예외가 발생함: {e}")

        exit("hwp가 생성되질 않아 종료 함.")

    def _close_hwp_windows() -> None:
        for proc in psutil.process_iter(["pid", "name"]):
            # 프로세스 이름이 Hwp.exe임.
            if "Hwp" not in proc._name:
                continue
            try:
                proc.terminate()
            except BaseException as e:  # noqa: F841
                pass

    def _empty_temp_cache() -> None:
        # cache flushing
        for path in Path(temp_dir_path).rglob("*"):
            try:
                path.unlink(missing_ok=True)
            except BaseException as e:  # noqa: F841
                pass

    _close_hwp_windows()
    hwp = _open_hwp_windows()

    max_retries = 5
    for data in tqdm(dataset, desc="extract_hwp"):
        doc_title = data["metadata"]["doc_title"].replace("/", "|")
        hwp_dir_path = base_path.joinpath(doc_title)

        hwp_num, pickle_num = len(list(hwp_dir_path.glob("*.hwp"))), len(list(hwp_dir_path.glob("*.pickle")))
        if pickle_num == hwp_num:
            continue

        _empty_temp_cache()

        for attempt in range(max_retries):
            try:
                # table_ls, mean_time = _extract_html_table(hwp_dir_path)
                _extract_html_table(hwp_dir_path)
                break
            except BaseException as e:
                logger.deebug(
                    f"attempt: {attempt}, hwp를 다시 생성함.",
                    f"디음과 같은 애러가 발생함: {e}",
                )

                # NOTE: 종종 아무 이유 없이 hwp가 종료되거나 연결이 끊어지는 경우가 있음. 그런 경우에 except로 빠져서 hwp 죽인뒤 다시 생성함.
                if "RPC 서버를 사용할 수 없습니다" in str(e):
                    raise RuntimeError("RPC 서버를 사용할 수 없습니다 애러가 발생.")

                _close_hwp_windows()
                _open_hwp_windows()
        else:
            # NOTE: for-else문, retry를 다 소진한 경우 무슨 짓을 해도 table을 추출할 수 없는 파일이기에
            #       hwp 프로세스를 죽이고, 다음 파일을 처리하도록 함.
            logger.debug(f"hwp_dir_path: {hwp_dir_path.stem}, 해당 파일에는 문제가 있음. 해당 파일은 건너 뜀.")

            _close_hwp_windows()
            _open_hwp_windows()

    hwp.close()
    logger.info("table extract from file is finish!!!!!")


def extract_image_from_html(example: Dataset, rank: int, base_path: Path) -> Dataset:
    def _open_chrome_windows(max_retries: int = 10) -> webdriver.Chrome:
        service = Service(executable_path=ChromeDriverManager().install())
        for retries in range(max_retries):
            try:
                driver = webdriver.Chrome(service=service)
                return driver
            except BaseException as e:
                _close_chrome_windows()
                print(f"driver {retries}회 실행 때 다음의 예외가 발생함: {e}")

        exit("driver가 생성되질 않아 종료 함.")

    def _close_chrome_windows() -> None:
        for proc in psutil.process_iter(["pid", "name"]):
            if "chrome" not in proc._name:
                continue
            try:
                proc.terminate()
            except BaseException as e:  # noqa: F841
                pass

    def _get_html_table_df(html: str) -> Tuple[pd.DataFrame, int]:
        text_filter_func = lambda cell: (  # noqa: E731
            only_ko_en_chr.sub("", cell.strip()).lower()
            if isinstance(cell, str)
            else only_ko_en_chr.sub("", str(cell))
        )
        nan_norm = lambda x: np.nan if x == "nan" else x  # noqa: E731

        html_table_df = pd.read_html(io.StringIO(html))[0]
        html_table_df = html_table_df.map(text_filter_func).map(nan_norm)

        df_nan_count = html_table_df.isna().sum().sum()
        html_table_df = html_table_df.fillna("")
        return (html_table_df, df_nan_count)

    def _find_matched_table(context: str, extract_table_ls: List[Table]) -> Optional[str]:
        original_table, original_nan_count = _get_html_table_df(context)

        matched_table_ls = list()
        threshold = 0.8
        for idx, table in enumerate(extract_table_ls):
            if not (table.html and (table.row, table.col) == original_table.shape):
                continue
            extract_table, extract_nan_count = _get_html_table_df(table.html)

            shape_flag = original_table.shape != (table.row, table.col)
            nan_flag = original_nan_count != extract_nan_count
            if nan_flag or shape_flag:
                "nan mismatch"
                continue

            # cell 100개 중에 nan이 90, 값이 있는 cell이 10일 때, 값이 있는 구간에선 다 틀렸는데 nan이 우연히 겹쳐서 threashold를 만족할 수 있음.
            # 미연에 방지하기 위해 nan을 계산하는 구간을 추가함.
            match_cell_num = (original_table == extract_table).sum().sum() - extract_nan_count
            match_percentage = match_cell_num / ((table.row * table.col) - original_nan_count)

            if match_percentage > threshold:
                matched_table_ls.append((match_percentage, table.html))

        if matched_table_ls:
            table = sorted(matched_table_ls, key=lambda table: table[0], reverse=True)[0]
            return table[1]
        return None

    def _get_image_from_web(driver, html_file_path) -> bytes:
        driver.get(f"file://{html_file_path.as_posix()}")

        element = WebDriverWait(driver, 10)
        element = element.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table")))

        # NOTE: element.screenshot으로 하면 이미지가 깨짐.
        result = driver.execute_cdp_cmd(
            "Page.captureScreenshot",
            {
                "captureBeyondViewport": True,
                "fromSurface": True,
                "clip": {
                    "x": element.location["x"],
                    "y": element.location["y"],
                    "width": element.size["width"],
                    "height": element.size["height"],
                    "scale": 1,
                },
            },
        )

        image = PIL_Image.open(io.BytesIO(base64.b64decode(result["data"])))
        return image

    # _close_chrome_windows()
    driver = _open_chrome_windows()

    col_num_ls, row_num_ls = list(), list()
    img_flag_ls, matched_finish_ls, image_finish_ls = list(), list(), list()
    for metadata, context in zip(example["metadata"], example["context"]):
        doc_title = metadata["doc_title"]
        hwp_dir_path = base_path.joinpath(doc_title.replace("/", "|"))

        image_bytes = None
        matched_table = None
        img_flag = False
        row_num, col_num = None, None
        for extract_table_pickle in hwp_dir_path.glob("*.pickle"):
            len_check = len(extract_table_pickle.as_posix())
            if len_check > 260:
                # 윈도우 최대 경로 길이는 260자임. 이거 넘어가면 파일이 있어도 파일 못찾는다고 애러 발생함.
                # 환경 변수로 max_len 지정할 수 있는데, 번거롭자너, 한잔해
                continue
            table_ls = pickle.loads(extract_table_pickle.read_bytes())
            try:
                matched_table = _find_matched_table(context, table_ls)

                df_table, _ = _get_html_table_df(matched_table)
                row_num, col_num = df_table.shape
            except BaseException as e:
                continue

            if not matched_table:
                continue

            matched_table_bs = BeautifulSoup(matched_table, features="lxml")
            for idx, p_tag in enumerate(matched_table_bs.find_all("p")):
                # 00000000은 색갈이 없는 공백이란 표현임. 그렇기 때문에 글자와 배경의 색갈이 겹칠 우려가 없음.
                p_tag["style"] = p_tag["style"].replace("background:#000000", "background:#00000000")
            matched_table = str(matched_table_bs)

            html_file_path = hwp_dir_path.joinpath(f"table-{rank}.html")
            html_file_path.write_text(matched_table, encoding="utf-8")

            max_retry = 10
            for retries in range(max_retry):
                try:
                    image_bytes = _get_image_from_web(driver, html_file_path)
                    break
                except BaseException as e:
                    driver = _open_chrome_windows()
                    continue

            html_file_path.unlink(missing_ok=True)
            img_flag = True
            break

        row_num_ls.append(row_num)
        col_num_ls.append(col_num)
        image_finish_ls.append(image_bytes)
        matched_finish_ls.append(matched_table)
        img_flag_ls.append(img_flag)

    driver.close()

    example["image"] = image_finish_ls
    example["original_context"] = matched_finish_ls
    example["img_flag"] = img_flag_ls

    example["col_num"] = col_num_ls
    example["row_num"] = row_num_ls

    return example


def augmentate_answer_to_gpt(example: Dataset) -> Dataset:
    def send_request_to_gpt(
        messages,
        model: str = "gpt-3.5-turbo-0125",
        retry: int = 10,
        error_interval_time: int = 10,
    ) -> str:
        for retries in range(retry):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "text"},
                )
                augmentate_answer = response.choices[0].message.content
                break
            except BaseException as e:
                logger.debug(f"{retries}회의 instruction 생성 중 {e}과 같은 애러가 발생해 다시 retry함.")
                sleep(error_interval_time)
        else:
            augmentate_answer = ""

        return augmentate_answer

    new_question_answer = list()
    for question_answer, context in zip(example["question_answer"], example["context"]):
        new_labels = list()
        html_table = str(BeautifulSoup(io.StringIO(context), "html.parser").find("table"))
        for labels in question_answer:
            messages = [
                {
                    "role": "user",
                    "content": f"# INSTURCTION: {GPT_INSTRUCTION}",
                },
                {"role": "user", "content": f"# TABLE: {html_table}"},
                {"role": "user", "content": labels["question"]},
                {"role": "assistant", "content": labels["answer"]},
                {"role": "user", "content": labels["question"]},
            ]
            augmentate_answer = send_request_to_gpt(messages)

            labels["augmentate_answer"] = augmentate_answer
            new_labels.append(labels)
        new_question_answer.append(new_labels)
    example["question_answer"] = new_question_answer

    return example


class DocumentTableVQAInst(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [BuilderConfig(name="default", version=_VERSION, description=_DESCRIPTION)]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self) -> DatasetInfo:
        logger.info(
            "원할한 프로그램 실행을 위해 해당 컴퓨터에 올라와 있는 모든 크롬과 클립보드의 권한을 가져감.",
            "이 데이터가 만들어지고 있는 동안 해당 컴퓨터에선 복사 & 붙여넣기, 크롬 사용이 원할하게 되지 않음.",
            "**특히 데이터 생성할 때 클립보드 권한 가져가서 복사 붙여넣기 안되니깐 진짜 명심하셈.**",
            "사양 적당히 좋은 서브 컴퓨터에서 돌리셈",
        )

        if self.config.name == "default":
            features = Features(
                {
                    "id": Value("int32"),
                    "image": Image(),
                    "context": Value("string"),
                    "original_context": Value("string"),
                    "conversations": [{"role": Value("string"), "content": Value("string")}],
                    "metadata": {
                        "class": Value("string"),
                        "code": Value("string"),
                        "created": Value("string"),
                        "doc_id": Value("int32"),
                        "doc_published": Value("int32"),
                        "doc_source": Value("string"),
                        "doc_title": Value("string"),
                        "col_num": Value("int32"),
                        "row_num": Value("int32"),
                    },
                }
            )
        else:
            raise NotImplementedError()
        return DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            version=_VERSION,
        )

    def aihub_downloader(self, destination_path: Path) -> None:
        aihub_id = os.getenv("AIHUB_ID", None)
        aihub_pass = os.getenv("AIHUB_PASS", None)

        if not aihub_id:
            raise ValueError(
                """AIHUB_ID가 지정되지 않았습니다. `os.environ["AIHUB_ID"]="your_id"`로 ID를 지정해 주세요"""
            )
        if not aihub_pass:
            raise ValueError(
                """AIHUB_PASS가 지정되지 않았습니다. `os.environ["AIHUB_PASS"]="your_pass"`로 ID를 지정해 주세요"""
            )

        response = requests.get(
            DOWNLOAD_URL,
            headers={"id": aihub_id, "pass": aihub_pass},
            params={"fileSn": "all"},
            stream=True,
        )

        if response.status_code == 502:
            raise BaseException(
                "다운로드 서비스는 홈페이지(https://aihub.or.kr)에서 신청 및 승인 후 이용 가능 합니다."
            )
        elif response.status_code != 200:
            raise BaseException(f"Download failed with HTTP status code: {response.status_code}")

        data_file = open(destination_path, "wb")
        downloaded_bytes = 0
        with tqdm(total=round(DATASET_SIZE * 1024**2)) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                data_file.write(chunk)
                downloaded_bytes += len(chunk)

                pbar.update(1)
                prefix = f"Downloaded (GB): {downloaded_bytes / (1024**3):.4f}/{DATASET_SIZE}"
                pbar.set_postfix_str(prefix)

        data_file.close()

    def concat_zip_part(self, unzip_dir: Path) -> None:
        part_glob = Path(unzip_dir).rglob("*.zip.part*")

        part_dict = dict()
        for part_path in part_glob:
            parh_stem = str(part_path.parent.joinpath(part_path.stem))

            if parh_stem not in part_dict:
                part_dict[parh_stem] = list()

            part_dict[parh_stem].append(part_path)

        for dst_path, part_path_ls in part_dict.items():
            with open(dst_path, "wb") as byte_f:
                for part_path in natsorted(part_path_ls):
                    byte_f.write(part_path.read_bytes())
                    os.remove(part_path)

    def _split_generators(self, dl_manager) -> List[SplitGenerator]:  # type: ignore
        cache_dir = Path(dl_manager.download_config.cache_dir)

        unzip_dir = cache_dir.joinpath(_DATANAME)
        tar_file = cache_dir.joinpath(f"{_DATANAME}.tar")

        if tar_file.exists():
            os.remove(tar_file)

        if not unzip_dir.exists():
            self.aihub_downloader(tar_file)

            with TarFile(tar_file, "r") as mytar:
                mytar.extractall(unzip_dir)
                os.remove(tar_file)

            self.concat_zip_part(unzip_dir)

        zip_path_ls = list(unzip_dir.rglob("*.zip"))

        train_path_ls = [path for path in zip_path_ls if "Training" in path.as_posix()]
        train_dw_base_path = cache_dir.joinpath("DocumentTableVQAInst", "train")
        train_dw_base_path.mkdir(parents=True, exist_ok=True)

        valid_path_ls = [path for path in zip_path_ls if "Validation" in path.as_posix()]
        valid_dw_base_path = cache_dir.joinpath("DocumentTableVQAInst", "valid")
        valid_dw_base_path.mkdir(parents=True, exist_ok=True)

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "filepath": train_path_ls,
                    "dw_base_path": train_dw_base_path,
                    "split": "train",
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "filepath": valid_path_ls,
                    "dw_base_path": valid_dw_base_path,
                    "split": "validation",
                },
            ),
        ]

    def _run_download_file_ps(self, dataset: Dataset, base_path: Path, max_retry: int = 10) -> None:
        for retries in range(1, max_retry):
            download_ps = mp.Process(
                name="hwp_download_ps",
                target=download_file_from_site,
                kwargs={"dataset": dataset, "base_path": base_path},
            )
            download_ps.start()
            download_ps.join()
            download_ps.terminate()

            if download_ps.exitcode != 0:
                continue

            break
        else:
            exit("실행횟수를 전부 소진, 프로그램을 종료함.")

    def _run_extract_table_ps(self, dataset: Dataset, base_path: Path, max_retry: int = 100) -> None:
        for retries in range(1, max_retry):
            # NOTE: 원격 프로시저를 실행하지 못했습니다. 애러를 피하기 위해 짠 코드.
            #       이유는 모르는데, 저 애러가 발생한 프로세스에선 hooker가 hwp에 attach를 못하더라?
            #       그래서 간단한 방법으로 ps를 다시 생성함. 원인이 뭔지 찾는 사람을 없을 것 같은데, 찾은 사람 있음 공유점
            #       근데 이 애러가 거의 1시간에 1번 꼴로 굉장히 자주 발생함. 그래서 여기만 retry가 100회 정도 됨.
            download_ps = mp.Process(
                name="hwp_extract_ps",
                target=extract_table_from_file,
                kwargs={"dataset": dataset, "base_path": base_path},
            )
            download_ps.start()
            download_ps.join()
            download_ps.terminate()

            # ps가 정상 종료하면 0으로 나옴. 그렇지 않으면 모두 재시작 하는 걸로
            if download_ps.exitcode != 0:
                continue

            break
        else:
            exit("실행횟수를 전부 소진, 프로그램을 종료함.")

    def _run_extract_image_ps(self, dataset: Dataset, base_path: Path) -> None:
        dataset = dataset.map(
            extract_image_from_html,
            num_proc=5,
            batched=True,
            batch_size=2000,
            with_rank=True,
            keep_in_memory=False,
            fn_kwargs={"base_path": base_path},
        )
        return dataset

    def _run_augmentate_answer(self, dataset: Dataset, base_path: Path) -> None:
        img_flag_ls = dataset["img_flag"]
        select_idx_ls = [idx for idx, img_flag in enumerate(img_flag_ls) if img_flag]

        dataset = dataset.select(select_idx_ls)
        dataset = dataset.map(
            augmentate_answer_to_gpt,
            num_proc=4,
            batched=True,
            keep_in_memory=True,
        )

        return dataset

    def _generate_examples(self, filepath: List[dict], dw_base_path: Path, split: str):
        label_zip_ls = [ZipFile(path) for path in filepath if "라벨링데이터" in path.as_posix()]

        label_zip = label_zip_ls[0].open(label_zip_ls[0].filelist[0]).read()
        dataset = json.loads(label_zip.decode("utf-8"))["data"]

        new_dataset = list()
        for data in dataset:
            paragraphs = data["paragraphs"][0]

            if len(data["paragraphs"]) != 1:
                continue

            label_ls = list()
            for qa in paragraphs["qas"]:
                label_ls.append(
                    {
                        "label_id": int(qa["question_id"]),
                        "is_impossible": qa["is_impossible"],
                        "question": qa["question"],
                        "answer": qa["answer"]["text"],
                        "answer_start": qa["answer"]["answer_start"],
                    }
                )

            data = {
                "id": int(paragraphs["context_id"]),
                "context": paragraphs["context"],
                "title": paragraphs["table_title"],
                "question_answer": label_ls,
                "metadata": {
                    "doc_id": int(data["doc_id"]),
                    "doc_title": data["doc_title"],
                    "doc_source": data["doc_source"],
                    "doc_published": int(data["doc_published"]),
                    "created": data["created"],
                    "class": data["doc_class"]["class"],
                    "code": data["doc_class"]["code"],
                },
            }
            new_dataset.append(data)

        dataset = Dataset.from_list(new_dataset)

        self._run_download_file_ps(dataset=dataset, base_path=dw_base_path)
        self._run_extract_table_ps(dataset=dataset, base_path=dw_base_path)
        dataset = self._run_extract_image_ps(dataset=dataset, base_path=dw_base_path)
        dataset = self._run_augmentate_answer(dataset=dataset, base_path=dw_base_path)

        for idx, data in enumerate(dataset):
            if not data["img_flag"]:
                continue

            data["question_answer"] = [
                question_answer for question_answer in data["question_answer"] if not question_answer["is_impossible"]
            ]
            conversations = list()
            fst_question_answer = data["question_answer"][0]
            fst_question = fst_question_answer["question"]
            fst_answer = fst_question_answer["augmentate_answer"]

            conversations.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": fst_question},
                    ],
                }
            )
            conversations.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": fst_answer}],
                }
            )

            for question_answer in data["question_answer"][1:]:
                question = {
                    "role": "user",
                    "content": [{"type": "text", "text": question_answer["question"]}],
                }
                answer = {
                    "role": "assistant",
                    "content": [{"type": "text", "text": question_answer["augmentate_answer"]}],
                }
                conversations.append(question)
                conversations.append(answer)

            for chat_idx, chat in enumerate(conversations):
                chat["content"] = json.dumps(chat["content"], ensure_ascii=False)
                conversations[chat_idx] = chat

            data["metadata"]["col_num"] = data["col_num"]
            data["metadata"]["row_num"] = data["row_num"]

            data = {
                "id": data["id"],
                "image": data["image"],
                "context": data["context"],
                "original_context": data["original_context"],
                "conversations": conversations,
                "metadata": data["metadata"],
            }

            yield (idx, data)


# NOTE: window는 무조건 spwan으로만 ps를 생성하기 때문에 이런 방식을 사용할 수 밖에 없음.
sys.modules["__main__"].__dict__["download_file_from_site"] = download_file_from_site
sys.modules["__main__"].__dict__["extract_table_from_file"] = extract_table_from_file
sys.modules["__main__"].__dict__["extract_image_from_html"] = extract_image_from_html
sys.modules["__main__"].__dict__["augmenate_answer_to_gpt"] = augmentate_answer_to_gpt

sys.modules["__main__"].__dict__["Table"] = Table
