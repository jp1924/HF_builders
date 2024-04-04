# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
from tarfile import TarFile
from zipfile import ZipFile, ZIP_DEFLATED
from typing import List

import requests
from datasets import (
    Audio,
    DatasetInfo,
    Features,
    GeneratorBasedBuilder,
    Split,
    SplitGenerator,
    Value,
)
from natsort import natsorted
from tqdm import tqdm

_LICENSE = """
AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다.
"""

_CITATION = None

_DESCRIPTION = """\
한국인의 일상 대화를 인식하고 음성을 문자로 실시간 변환하는 AI개발용 대화 한국어 음성 데이터
"""


DATASET_KEY = "130"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/{DATASET_KEY}.do"
_HOMEPAGE = f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"

_VERSION = "1.3.0"
_DATANAME = "KoreaSpeech"

subject = {
    "01": "방송",
    "02": "취미",
    "03": "일상안부",
    "04": "기술",
    "05": "생활",
    "06": "날씨",
    "07": "경제",
    "08": "놀이",
    "09": "쇼핑",
}
topic = {
    "01": {
        "01": "드라마",
        "02": "영화",
        "03": "K-POP",
        "04": "시사교양",
        "05": "예능",
        "06": "연예인",
        "07": "회화",
        "08": "다큐",
        "09": "뉴스",
        "10": "스포츠",
        "11": "만화",
        "12": "여행",
        "13": "건강",
        "14": "역사",
        "15": "교육",
        "99": "기타",
    },
    "02": {
        "01": "운동",
        "02": "공연",
        "03": "낚시",
        "04": "게임",
        "05": "여행",
        "06": "그림",
        "07": "음악",
        "08": "등산",
        "09": "독서",
        "10": "사진",
        "11": "음식",
        "12": "전시회",
        "13": "자동차",
        "99": "기타",
    },
    "03": {
        "01": "자기소개",
        "02": "거주지정보",
        "03": "이성친구",
        "04": "학교생활",
        "05": "회사생활",
        "06": "기념일",
        "07": "안부인사",
        "08": "코로나",
        "99": "기타",
    },
    "04": {
        "01": "4차산업",
        "02": "스마트폰",
        "03": "IT동향",
        "04": "인공지능",
        "05": "기술용어",
        "06": "자동차",
        "07": "게임",
        "99": "기타",
    },
    "05": {
        "01": "형제",
        "02": "가족",
        "03": "생계",
        "04": "농사",
        "05": "밭일",
        "06": "소일거리",
        "07": "직장생활",
        "08": "추억",
        "09": "반려동물",
        "10": "음식",
        "11": "조리",
        "12": "건강",
        "99": "기타",
    },
    "06": {
        "01": "계절",
        "02": "황사",
        "03": "미세먼지",
        "04": "악취",
        "05": "온도",
        "06": "장마",
        "07": "폭설",
        "08": "혹서기",
        "09": "혹한기",
        "10": "눈",
        "11": "비",
        "12": "안개",
        "99": "기타",
    },
    "07": {
        "01": "부동산",
        "02": "주식",
        "03": "경제지표",
        "04": "재테크",
        "99": "기타",
    },
    "08": {
        "01": "유치원생활",
        "02": "친구",
        "03": "엄마아빠",
        "04": "장난감",
        "05": "선생님",
        "99": "기타",
    },
    "09": {
        "01": "의류",
        "02": "전자기기",
        "03": "생활용품",
        "04": "악기",
        "05": "식품",
        "06": "소모품",
        "99": "기타",
    },
}
gender = {
    "M": "남",
    "F": "여",
    "X": "",
}
generation = {
    "C": "유아",
    "T": "청소년",
    "A": "일반성인",
    "S": "고령층",
    "Z": "기타",
}
location = {
    "1": "서울/경기",
    "2": "강원",
    "3": "충청",
    "4": "경상",
    "5": "전라",
    "6": "제주",
    "9": "기타",
}
dialect = {
    "1": "서울/경기",
    "2": "강원",
    "3": "충청",
    "4": "경상",
    "5": "전라",
    "6": "제주",
    "9": "기타",
}
source = {
    "1": "방송",
    "2": "제작",
    "3": "크라우드소싱",
    "9": "기타",
}
quality = {
    "1": "정상",
    "2": "노이즈",
    "3": "잡음",
    "4": "원거리",
}


class KoreaSpeech(GeneratorBasedBuilder):
    def _info(self) -> DatasetInfo:
        features = Features(
            {
                "audio": Audio(16000),
                "sentence": Value("string"),
                "id": Value("string"),
                "meta": {
                    "original": Value("string"),
                    "start": Value("string"),
                    "end": Value("string"),
                    "length": Value("string"),
                    "subject": Value("string"),
                    "topic": Value("string"),
                    "gender": Value("string"),
                    "generation": Value("string"),
                    "location": Value("string"),
                    "dialect": Value("string"),
                    "source": Value("string"),
                    "quality": Value("string"),
                },
            }
        )
        return DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            version=_VERSION,
        )

    def aihub_downloader(self, recv_path: Path) -> None:
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

        if response.status_code != 200:
            raise BaseException(f"Download failed with HTTP status code: {response.status_code}")

        with open(recv_path, "wb") as file:
            # chunk_size는 byte수
            for chunk in tqdm(response.iter_content(chunk_size=1024)):
                file.write(chunk)

    def concat_zip_part(self, unzip_dir: Path) -> None:
        part_glob = Path(unzip_dir).rglob("*.tar.gz.part*")

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

    def transfer_tar_to_zip(self, zip_file_path) -> None:
        # tar file로 불러들이니깐 너무 오래 걸림.
        for x in tqdm(zip_file_path, desc="transfer_to_zip"):
            tarf = TarFile.open(name=x, mode="r|gz")
            zip_path = x.parent.joinpath(f"""{x.name.replace(".tar.gz", "")}.zip""")
            zipf = ZipFile(file=zip_path, mode="a", compression=ZIP_DEFLATED)
            for m in tqdm(tarf):
                if m.isdir():
                    continue
                f = tarf.extractfile(m)
                fl = f.read()
                fn = m.name
                zipf.writestr(fn, fl)
            tarf.close()
            zipf.close()
            os.remove(x)

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
            self.transfer_tar_to_zip(list(unzip_dir.rglob("*.tar.gz")))

        zip_file_path = list(unzip_dir.rglob("*.zip"))
        train_split = [x for x in zip_file_path if "Training" in str(x)]
        valid_split = [x for x in zip_file_path if "Validation" in str(x)]

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "filepath": train_split,
                    "split": "train",
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "filepath": valid_split,
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(self, filepath: List[Path], split: str):
        # 원천데이터 안에 meta, label이 전부 들어가 있음.
        source_ls = [ZipFile(x) for x in filepath if "원천데이터" in str(x)]
        source_ls = natsorted(source_ls, key=lambda x: x.filename)

        idx = 0
        for source_zip in source_ls:
            source_info_ls = [x for x in source_zip.filelist if not x.is_dir()]

            source_dict = dict()
            for source_info in source_info_ls:
                _id = source_info.filename.split("/")[-1]
                _id = _id.split(".")[0]
                if _id not in source_dict:
                    source_dict[_id] = list()
                source_dict[_id].append(source_info)

            source_dict = {
                _id: {info.filename.split(".")[-1]: info for info in info_ls}
                for _id, info_ls in source_dict.items()
            }
            for _, source_info in source_dict.items():
                # 여기 바꿔야 함.
                audio = source_zip.open(source_info["wav"]).read()
                meta = json.loads(source_zip.open(source_info["json"]).read().decode("utf-8"))
                sentence = source_zip.open(source_info["txt"]).read().decode("utf-8")

                metadata = meta.pop("metadata")
                _id, metadata = tuple(metadata.split("_"))

                try:
                    meta["subject"] = subject[metadata[0:2]]
                    meta["topic"] = topic[metadata[0:2]][metadata[2:4]]
                    meta["gender"] = gender[metadata[4:5]]
                    meta["generation"] = generation[metadata[5:6]]
                    meta["location"] = location[metadata[6:7]]
                    meta["dialect"] = dialect[metadata[7:8]]
                    meta["source"] = source[metadata[8:9]]
                    meta["quality"] = quality[metadata[9:10]]
                except:
                    # NOTE: 0599FA삭제삭제31 같은 데이터가 있음.
                    meta["subject"] = ""
                    meta["topic"] = ""
                    meta["gender"] = ""
                    meta["generation"] = ""
                    meta["location"] = ""
                    meta["dialect"] = ""
                    meta["source"] = ""
                    meta["quality"] = ""

                data = {
                    "id": _id,
                    "sentence": sentence,
                    "audio": audio,
                }
                data["meta"] = meta

                yield (idx, data)
                idx += 1  #
