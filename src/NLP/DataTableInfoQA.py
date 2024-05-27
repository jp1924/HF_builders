# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
from tarfile import TarFile
from typing import List
from zipfile import ZipFile

import requests
from datasets import (
    BuilderConfig,
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

_DESCRIPTION = (
    """테이블이 포함된 일반 문서 내에서 표 내의 특정 값을 탐색하기 위한 기계학습용 질의어와 정답 세트 데이터"""
)


DATASET_KEY = "71565"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/{DATASET_KEY}.do"
_HOMEPAGE = (
    f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"
)

_VERSION = "1.1.0"
_DATANAME = "DataTableInfoQA"
DATASET_SIZE = 0.21531


class DataTableInfoQA(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [BuilderConfig(name="default", version=_VERSION, description=_DESCRIPTION)]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self) -> DatasetInfo:
        if self.config.name == "default":
            features = Features(
                {
                    "id": Value("string"),
                    "context": Value("string"),
                    "title": Value("string"),
                    "labels": [
                        {
                            "id": Value("string"),
                            "is_impossible": Value("bool"),
                            "question": Value("string"),
                            "answer": Value("string"),
                            "answer_start": Value("int32"),
                        },
                    ],
                    "metadata": {
                        "doc_id": Value("string"),
                        "doc_title": Value("string"),
                        "doc_source": Value("string"),
                        "doc_published": Value("int32"),
                        "created": Value("string"),
                        "class": Value("string"),
                        "code": Value("string"),
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

    def _generate_examples(self, filepath: List[dict], split: str):
        label_zip = [ZipFile(x) for x in filepath if "라벨링데이터" in str(x)][0]

        label_zip = label_zip.open(label_zip.filelist[0]).read()
        labels = json.loads(label_zip.decode("utf-8"))

        for idx, data in enumerate(labels["data"]):
            if len(data["paragraphs"]) != 1:
                raise ValueError("paragraphs 무조건 하나 밖에 없음.")

            paragraphs = data["paragraphs"][0]

            label_ls = list()
            for qa in paragraphs["qas"]:
                label_ls.append(
                    {
                        "id": qa["question_id"],
                        "is_impossible": qa["is_impossible"],
                        "question": qa["question"],
                        "answer": qa["answer"]["text"],
                        "answer_start": qa["answer"]["answer_start"],
                    }
                )

            data = {
                "context": paragraphs["context"],
                "id": paragraphs["context_id"],
                "title": paragraphs["table_title"],
                "labels": label_ls,
                "metadata": {
                    "doc_id": data["doc_id"],
                    "doc_title": data["doc_title"],
                    "doc_source": data["doc_source"],
                    "doc_published": data["doc_published"],
                    "created": data["created"],
                    "class": data["doc_class"]["class"],
                    "code": data["doc_class"]["code"],
                },
            }

            yield (idx, data)
