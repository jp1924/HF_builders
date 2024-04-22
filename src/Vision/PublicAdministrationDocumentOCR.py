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
    Image,
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
공공 행정 문서에 특화된 문자 인식 AI 모델을 개발하기 위한 공공 행정 문서 이미지 데이터
"""


DATASET_KEY = "88"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/{DATASET_KEY}.do"
_HOMEPAGE = f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"

_VERSION = "1.2.0"
_DATANAME = "PublicAdministrationDocumentOCR"
DATASET_SIZE = 384.89


class PublicAdministrationDocumentOCR(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [BuilderConfig(name="default", version=_VERSION)]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self) -> DatasetInfo:
        features = Features(
            {
                "image": Image(),
                "meta_data": [
                    {
                        "image.make.code": Value("string"),
                        "image.make.year": Value("string"),
                        "image.category": Value("string"),
                        "image.width": Value("int32"),
                        "image.height": Value("int32"),
                        "image.file.name": Value("string"),
                        "image.create.time": Value("string"),
                    }
                ],
                "objects": [
                    {
                        "id": Value("int32"),
                        "text": Value("string"),
                        "bbox": (Value("int32")),
                        "meta": {
                            "type": Value("string"),
                            "text_type": Value("string"),
                        },
                    }
                ],
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

    def _generate_examples(self, filepath: List[Path], split: str):
        source_ls = [ZipFile(x) for x in filepath if "[원천]" in str(x)]
        label_ls = [ZipFile(x) for x in filepath if "[라벨]" in str(x)]
        info_replacer = lambda info, key: info.filename.split("/")[-1].replace(key, "")

        source_dict = dict()
        source_check_count = 0
        for source_zip in source_ls:
            info_ls = [info for info in source_zip.filelist if not info.is_dir()]
            info_dict = {
                info_replacer(info, ".jpg"): source_zip.open(info).read() for info in info_ls
            }

            source_check_count += len(info_ls)
            source_dict.update(info_dict)

        if len(source_dict) != source_check_count:
            raise ValueError("데이터 개수가 맞지 않음.")

        label_dict = dict()
        label_check_count = 0
        for label_zip in label_ls:
            info_ls = [info for info in label_zip.filelist if not info.is_dir()]
            info_dict = {
                info_replacer(info, ".json"): label_zip.open(info).read() for info in info_ls
            }

            label_check_count += len(info_ls)
            label_dict.update(info_dict)

        if len(label_dict) != label_check_count:
            raise ValueError()

        if source_check_count != label_check_count:
            raise ValueError()

        idx_count = 0
        for file_name, image_byte in source_dict.items():
            if file_name not in label_dict:
                print()
                continue

            label_bytes = label_dict[file_name]
            encoding_type = json.detect_encoding(label_bytes)
            label = json.loads(label_bytes.decode(encoding_type))

            images = label.pop("images")
            old_bbox_ls = label.pop("annotations")

            new_bbox_ls = list()
            for bbox in old_bbox_ls:
                # [x_min, y_min, x_max, y_max]
                coordinate = (
                    bbox["annotation.bbox"][0],
                    bbox["annotation.bbox"][1],
                    bbox["annotation.bbox"][2] + bbox["annotation.bbox"][0],
                    bbox["annotation.bbox"][3] + bbox["annotation.bbox"][1],
                )
                meta = {"type": bbox["annotation.type"], "text_type": bbox["annotation.ttype"]}

                new_bbox = {
                    "id": bbox["id"],
                    "text": bbox["annotation.text"],
                    "bbox": coordinate,
                    "meta": meta,
                }
                new_bbox_ls.append(new_bbox)

            dataset = {
                "objects": new_bbox_ls,
                "image": image_byte,
                "meta_data": images,
            }
            yield (idx_count, dataset)
            idx_count += 1
