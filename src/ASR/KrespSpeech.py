# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
from tarfile import TarFile
from typing import List
from zipfile import ZipFile

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
다양한 매장과 공간의 키오스크 등에서 주문, 검색, 조작 및 고객 응대 하는 한국어 음성 데이터
"""


DATASET_KEY = "87"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/{DATASET_KEY}.do"
_HOMEPAGE = (
    f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"
)

_VERSION = "1.2.0"
_DATANAME = "KrespSpeech"


class KrespSpeech(GeneratorBasedBuilder):
    def _info(self) -> DatasetInfo:
        features = Features(
            {
                "audio": Audio(16000),
                "sentence": Value("string"),
                "id": Value("string"),
                "dataSet": {
                    "version": Value("string"),
                    "date": Value("date32"),
                    "typeInfo": {
                        "category": Value("string"),
                        "subcategory": Value("string"),
                        "place": Value("string"),
                        "speakers": [
                            {
                                "id": Value("string"),
                                "gender": Value("string"),
                                "type": Value("string"),
                                "age": Value("string"),
                                "residence": Value("string"),
                            }
                        ],
                        "inputType": Value("string"),
                    },
                    "dialogs": [
                        {
                            "speaker": Value("string"),
                            "audioPath": Value("string"),
                            "textPath": Value("string"),
                        }
                    ],
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
        source_ls = [ZipFile(x) for x in filepath if "원천데이터" in str(x)]
        label_ls = [ZipFile(x) for x in filepath if "라벨링데이터" in str(x)]

        source_ls = natsorted(source_ls, key=lambda x: x.filename)
        label_ls = natsorted(label_ls, key=lambda x: x.filename)

        idx = 0
        for source_zip, label_zip in zip(source_ls, label_ls):
            source_zip_file_info = [x for x in source_zip.filelist if not x.is_dir()]
            label_zip_file_info = [x for x in label_zip.filelist if not x.is_dir()]

            label_dict = dict()
            for label_info in label_zip_file_info:
                _id = label_info.filename.split("/")[-2]
                if _id not in label_dict:
                    label_dict[_id] = list()
                label_dict[_id].append(label_info)

            source_dict = dict()
            for source_info in source_zip_file_info:
                _id = source_info.filename.split("/")[-2]
                source_dict[_id] = source_info

            label_dict = {
                _id: {info.filename.split(".")[-1]: info for info in info_ls} for _id, info_ls in label_dict.items()
            }

            for _id, source_info in source_dict.items():
                label = label_dict[_id]

                sentence = label_zip.open(label["txt"]).read().decode("utf-8")
                meta = json.loads(label_zip.open(label["json"]).read().decode("utf-8"))
                audio = source_zip.open(source_info).read()

                data = {
                    "id": _id,
                    "sentence": sentence,
                    "audio": audio,
                }
                data.update(meta)

                yield (idx, data)

                idx += 1
