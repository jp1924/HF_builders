# -*- coding: utf-8 -*-
import io
import json
import os
import random
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
from datasets.config import DEFAULT_MAX_BATCH_SIZE
from natsort import natsorted
from PIL import Image as PIL_Image
from tqdm import tqdm

from transformers import set_seed
from transformers.trainer_pt_utils import get_length_grouped_indices


set_seed(42)

_LICENSE = """
AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다.
"""

_CITATION = None

_DESCRIPTION = """\
한국 상황을 잘 설명할 수 있는 한국형 객체인식 데이터셋 구축하기 위해 300만장의 이미지로부터 한글/영문 각 10개의 설명문을 도출함
"""


DATASET_KEY = "71454"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/{DATASET_KEY}.do"
_HOMEPAGE = (
    f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"
)

_VERSION = "1.1.0"
_DATANAME = "KoreanVisionDataforImageDescriptionSentenceExtractionandGeneration"
DATASET_SIZE = 1464.32


class KoreanVisionDataforImageDescriptionSentenceExtractionandGeneration(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="object",
            version=_VERSION,
            data_dir="object",
            description="객체 데이터",
        ),
        BuilderConfig(
            name="caption",
            version=_VERSION,
            data_dir="caption",
            description="캡션 데이터",
        ),
    ]

    DEFAULT_CONFIG_NAME = "caption"

    def _info(self) -> DatasetInfo:
        if self.config.name == "caption":
            features = Features(
                {
                    "id": Value("int32"),
                    "image": Image(),
                    "caption": Value("string"),
                    "caption_ls": [Value("string")],
                    "category": Value("string"),
                    "en_caption": [Value("string")],
                    "metadata": {
                        "description": Value("string"),
                        "version": Value("string"),
                        "data_year": Value("string"),
                        "main_category": Value("string"),
                        "width": Value("int32"),
                        "height": Value("int32"),
                        "file_name": Value("string"),
                        "supercategory": Value("string"),
                    },
                }
            )
        elif self.config.name == "object":
            features = Features(
                {
                    "image": Image(),
                    "id": Value("string"),
                    "height": Value("int16"),
                    "width": Value("int16"),
                    "file_name": Value("string"),
                    "categories": [
                        {
                            "supercategory": Value("string"),
                            "id": Value("string"),
                            "name": Value("string"),
                        }
                    ],
                    "info": {
                        "description": Value("string"),
                        "version": Value("string"),
                        "year": Value("string"),
                        "main_category": Value("string"),
                    },
                    "annotations": [
                        {
                            "id": Value("int16"),
                            "image_id": Value("int16"),
                            "category_id": Value("int16"),
                            "iscrowd": Value("int8"),
                            "bbox": [Value("float32")],
                            "area": Value("float32"),
                        }
                    ],
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

    def _caption_generate_examples(self, filepath: List[Path], split: str):
        source_zip_ls = [ZipFile(path) for path in filepath if "원천데이터" in path.as_posix()]
        source_zip_ls = [source_zip for source_zip in source_zip_ls if "(영상)" not in source_zip.filename]
        source_zip_ls = natsorted(source_zip_ls, key=lambda path: path.filename)

        label_zip_ls = [ZipFile(path) for path in filepath if "라벨링데이터" in path.as_posix()]
        label_zip_ls = [label_zip for label_zip in label_zip_ls if "(영상)" not in label_zip.filename]
        label_zip_ls = [label_zip for label_zip in label_zip_ls if "_라벨링_데이터_" in label_zip.filename]
        label_zip_ls = natsorted(label_zip_ls, key=lambda path: path.filename)

        # # /IMG_1580162_air_conditioner(air_conditioner(ceiling)).jpg 이레 생겨서 슬라이싱 함.
        source_info_dict = {
            file_info.filename[1:]: (file_info, Path(source_zip.filename).stem)
            for source_zip in tqdm(source_zip_ls)
            for file_info in source_zip.filelist
            if not file_info.is_dir()
        }
        source_zip_dict = {Path(source_zip.filename).stem: source_zip for source_zip in source_zip_ls}
        label_zip_dict = {Path(label_zip.filename).stem: label_zip for label_zip in label_zip_ls}
        resolution_ls, label_ls = list(), list()

        for label_zip in tqdm(label_zip_ls):
            for label_zip_info in label_zip.filelist:
                label = json.loads(label_zip.open(label_zip_info).read().decode("utf-8"))
                image_info = label["images"][0]

                label_ls.append((label_zip_info, Path(label_zip.filename).stem))
                resolution_ls.append(image_info["height"] * image_info["width"])

        resolution_ls = get_length_grouped_indices(
            resolution_ls,
            batch_size=1,
            mega_batch_mult=DEFAULT_MAX_BATCH_SIZE,
        )

        for idx, label_idx in enumerate(resolution_ls):
            label_zip_info, label_zip_name = label_ls[label_idx]
            label = json.loads(label_zip_dict[label_zip_name].open(label_zip_info).read().decode("utf-8"))
            image_info = label["images"][0]

            zip_file_info, source_zip_name = source_info_dict[image_info["file_name"]]
            source_zip = source_zip_dict[source_zip_name]

            try:
                img_io = io.BytesIO(source_zip.open(zip_file_info).read())
                image = PIL_Image.open(img_io)
                image = image.convert("RGB")
            except BaseException:
                continue

            caption_ko_ls = [x["korean"] for x in label["annotations"]]
            caption_en_ls = [x["english"] for x in label["annotations"]]

            category = random.choice(label["categories"])

            label["info"]["width"] = image_info["width"]
            label["info"]["height"] = image_info["height"]
            label["info"]["file_name"] = image_info["file_name"]
            label["info"]["supercategory"] = category["supercategory"]

            data = {
                "id": int(image_info["id"]),
                "image": image,
                "caption": random.choice(caption_ko_ls),
                "caption_ls": caption_ko_ls,
                "category": category["name"],
                "en_caption": caption_en_ls,
                "metadata": label["info"],
            }

            yield (idx, data)

    def _generate_examples(self, **kwargs):
        if self.config.name == "caption":
            return self._caption_generate_examples(**kwargs)
        elif self.config.name == "object":
            return self._object_generate_examples(**kwargs)
        else:
            raise NotImplementedError()
