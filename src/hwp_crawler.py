import ctypes
import io
import json
import multiprocessing as mp
import pickle
import platform
import re
from dataclasses import dataclass
from multiprocessing import Queue
from pathlib import Path
from time import sleep
from typing import List
from urllib.parse import quote

import numpy as np
import pandas as pd
import win32clipboard as clipboard
from bs4 import BeautifulSoup
from datasets import load_dataset
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager


if platform.system() != "Windows":
    msg = "이 코드는 무조건 아레 한글이 설치된 windows에서만 동작이 가능합니다!"
    raise ValueError(msg)

only_ko_en_chr = re.compile("[^가-힣A-Za-z0-9]")


@dataclass
class Table:
    html: str
    row: int
    col: int


# https://stackoverflow.com/questions/101128/how-do-i-read-text-from-the-windows-clipboard-in-python/23285159#23285159
def get_win32_modules():
    kernel32 = ctypes.windll.kernel32
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    user32 = ctypes.windll.user32
    user32.GetClipboardData.restype = ctypes.c_void_p
    return user32, kernel32


def download_hwp_file(driver: "WebDriver", doc_title: str, doc_published: str) -> None:
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
        [
            tag.find_element(By.CSS_SELECTOR, "a.btn_mdl").click()
            for tag in driver.find_elements(By.ID, "epDialog")
        ]
        tag = "table.table_list_attchFileTbl"
        table_list = driver.find_element(By.CSS_SELECTOR, tag).find_elements(By.TAG_NAME, "a")
        [herf_link.click() for herf_link in table_list if "hwp" in herf_link.text]
        sleep(1)

        driver.back()
        driver.back()


def get_html_table_from_clipboard() -> str:
    while True:
        try:
            clipboard.OpenClipboard(0)
            format_id = clipboard.EnumClipboardFormats(0)
            break
        except:
            pass

    soup = None
    format_id_ls = list()
    while format_id:
        format_id = clipboard.EnumClipboardFormats(format_id)
        if not clipboard.IsClipboardFormatAvailable(format_id):
            continue
        format_id_ls.append(format_id)
        try:
            text = clipboard.GetClipboardData(format_id)
        except:
            format_id_ls
            continue
        if isinstance(text, bytes):
            if b"html" not in text:
                continue
            # ignore를 하지 않으면 애러 발생함.
            text = text.decode("utf-8", errors="ignore")
            soup = BeautifulSoup(text, "html.parser").find("html")
            format_id_ls.append(soup)
            # break

    clipboard.EmptyClipboard()
    clipboard.CloseClipboard()

    # 아무 정보도 없는 데이터가 추출될 때가 있음.
    if not soup:
        return ""
    # 정작 html을 추출 되었지만 중요한 table 테그가 없는 경우가 있음.
    elif not soup.find("table"):
        return ""

    return soup.encode().decode("utf-8")


def koneps_crawler(queue: Queue) -> None:
    def set_download_directory(driver, download_path):
        params = {"behavior": "allow", "downloadPath": download_path}
        driver.execute_cdp_cmd("Page.setDownloadBehavior", params)

    save_dir_path = Path(r"C:\Users\42Maru\Documents\extractor\table_hwp_dir")
    driver = webdriver.Chrome(service=service)
    dataset = load_dataset("jp1924/DataTableInfoQA", split="train")
    dataset = dataset.select(range(10))
    for data in tqdm(dataset):
        doc_title = data["metadata"]["doc_title"]
        doc_source = data["metadata"]["doc_source"]
        doc_published = str(data["metadata"]["doc_published"])

        if doc_source != "나라장터":
            continue

        enc_title = quote(doc_title, encoding="euc-kr")
        koneps_url = f"https://www.g2b.go.kr:8340/search.do?category=TGONG&kwd={enc_title}"
        driver.get(koneps_url)

        save_path = save_dir_path.joinpath(doc_title.replace("/", "|"))
        set_download_directory(driver, rf"{str(save_path)}")

        sleep(0.5)
        download_hwp_file(driver, doc_title, doc_published)

        queue.put((data["context"], save_path))
        save_path
    queue.put("is_end_packet")
    driver.close()
    print("hwp file download finish!!!!")


def html_table_extract_from_hwp(queue: Queue) -> None:
    def extract_table(hwp_dir_path) -> List[Table]:
        bid_notice_dir = list()
        for hwp_file_path in hwp_dir_path.glob("*.hwp"):
            one_file_table_ls = list()
            hwp.open(hwp_file_path.as_posix())
            ctrl = hwp.HeadCtrl
            while ctrl:
                if ctrl.UserDesc != "표":
                    ctrl = ctrl.Next
                    continue
                hwp.select_ctrl(ctrl)
                hwp.Copy()
                # sleep(0.25)
                html = get_html_table_from_clipboard()
                if not ((hwp.ShapeObjTableSelCell() and hwp.is_cell()) and html):
                    ctrl = ctrl.Next
                    continue

                table = Table(html=html, col=hwp.get_col_num(), row=hwp.get_row_num())
                one_file_table_ls.append(table)
                ctrl = ctrl.Next

            bid_notice_dir.extend(one_file_table_ls)

            pickle_save_path = hwp_file_path.parent.joinpath(f"{hwp_file_path.stem}.pickle")
            pickle_save_path.write_bytes(pickle.dumps(one_file_table_ls))

        return bid_notice_dir

    from pyhwpx import Hwp

    hwp = Hwp()
    bid_notice_ls = list()
    table_chack_num = 0
    skip_count_num = 0
    while True:
        if not queue.qsize():
            continue
        data = queue.get()
        if data == "is_end_packet":
            break

        context, hwp_dir_path = data

        if not list(hwp_dir_path.glob("*.hwp")):
            continue

        current_retry = 0
        while True:
            try:
                table_ls = extract_table(hwp_dir_path)
                break
            except:
                del hwp
                hwp = Hwp()
                current_retry += 1
                print(f"retry-{current_retry}")

            if current_retry == 0:
                exit()

        text_filter_func = lambda cell: (
            only_ko_en_chr.sub("", cell.strip()).lower()
            if isinstance(cell, str)
            else only_ko_en_chr.sub("", str(cell))
        )
        nan_norm = lambda x: np.nan if x == "nan" else x

        simple_table = pd.read_html(io.StringIO(context))[0]
        simple_table = simple_table.map(text_filter_func).map(nan_norm)
        simple_table_nan_num = simple_table.isna().sum().sum()
        simple_table = simple_table.fillna("")

        final_output = [table for table in table_ls if (table.row, table.col) == simple_table.shape]
        if not final_output:
            skip_count_num += 1
            continue

        threshold = 0.8
        for idx, table in enumerate(final_output):
            original_table = pd.read_html(io.StringIO(table.html))[0]
            if original_table.shape != (table.row, table.col):
                continue
            original_table = original_table.map(text_filter_func).map(nan_norm)
            original_table_nan_num = original_table.isna().sum().sum()
            original_table = original_table.fillna("")

            shape_flag = original_table.shape != (table.row, table.col)
            nan_flag = simple_table_nan_num != original_table_nan_num
            if nan_flag or shape_flag:
                # breakpoint()
                "nan mismatch"
                continue

            # cell 100개 중에 nan이 90, 값이 있는 cell이 10일 때, 값이 있는 구간에선 다 틀렸는데 nan이 우연히 겹쳐서 threashold를 만족할 수 있음.
            # 미연에 방지하기 위해 nan을 계산하는 구간을 추가함.
            match_cell_num = (simple_table == original_table).sum().sum() - original_table_nan_num
            match_percentage = match_cell_num / ((table.row * table.col) - simple_table_nan_num)

            if match_percentage > threshold:
                hwp_dir_path.joinpath(f"{len(bid_notice_ls)}-matched_table.json").write_text(
                    json.dumps({context: table.html}, ensure_ascii=False),
                    encoding="utf-8",
                )
                bid_notice_ls.append({context: table.html})
                break
        table_chack_num += 1
    hwp.close()
    print("table extract finish!!!!!")
    return "is end!"


# service = Service(executable_path=ChromeDriverManager().install())
# user32, kernel32 = get_win32_modules()


def main():
    queue = Queue()  # 공유 멤

    crawler_ps = mp.Process(name="koneps_crawler_ps", target=koneps_crawler, args=(queue,))
    extractor_ps = mp.Process(
        name="table_extractor_ps", target=html_table_extract_from_hwp, args=(queue,)
    )

    crawler_ps.start()
    extractor_ps.start()

    try:
        crawler_ps.join()
        extractor_ps.join()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Terminating processes.")
        crawler_ps.terminate()
        extractor_ps.terminate()
    except Exception as e:
        print(f"An exception occurred: {e}")
        crawler_ps.terminate()
        extractor_ps.terminate()


if "__main__" in __name__:
    main()
