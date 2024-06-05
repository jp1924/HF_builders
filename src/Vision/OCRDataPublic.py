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
일반 국민 생활과 밀접한 관련성이 높은 지방자치단체 (창원특례시, 김해시)와 외교 용어가 다수 포함되어있는 외교사료관 공공문서를 수집, 가공하여, 문서에 포함되어있는 다양한 문자 유형(인쇄체, 타자체, 수기 등)의 OCR 문자 인식 기술개발을 위한 인공지능 학습용 데이터셋
"""


DATASET_KEY = "71299"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/{DATASET_KEY}.do"
_HOMEPAGE = (
    f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"
)

_VERSION = "1.1.0"
_DATANAME = "OCRDataPublic"
DATASET_SIZE = 150.94


class OCRDataPublic(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [BuilderConfig(name="default", version=_VERSION)]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self) -> DatasetInfo:
        features = Features(
            {
                "image": Image(),
                "meta_data": {
                    "object_recognition": Value("int32"),
                    "text_language": Value("int32"),
                    "category": Value("int32"),
                    "identifier": Value("string"),
                    "label_path": Value("string"),
                    "name": Value("string"),
                    "src_path": Value("string"),
                    "type": Value("string"),
                    "acquisition_location": Value("int32"),
                    "data_captured": Value("string"),
                    "dpi": Value("int32"),
                    "group": Value("int32"),
                    "height": Value("int32"),
                    "width": Value("int32"),
                    "writing_style": Value("int32"),
                    "year": Value("int32"),
                },
                "objects": [
                    {
                        "id": Value("int32"),
                        "text": Value("string"),
                        "bbox": [Value("int32")],
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
        source_ls = [ZipFile(x) for x in filepath if "원천데이터" in str(x)]
        label_ls = [ZipFile(x) for x in filepath if "라벨링데이터" in str(x)]

        source_ls = natsorted(source_ls, key=lambda x: x.filename)
        label_ls = natsorted(label_ls, key=lambda x: x.filename)

        info_replacer = lambda info, key: info.filename.split("/")[-1].replace(key, "")

        idx_count = 0
        for image_zip, label_zip in zip(source_ls, label_ls):
            image_file_info = [x for x in image_zip.filelist if not x.is_dir()]
            label_file_info = [x for x in label_zip.filelist if not x.is_dir()]

            image_file_info = natsorted(image_file_info, key=lambda x: x.filename)
            label_file_info = natsorted(label_file_info, key=lambda x: x.filename)

            pair_ls = list()
            for label_info, image_info in zip(label_file_info, image_file_info):
                label_name = info_replacer(label_info, ".json")
                image_name = info_replacer(image_info, ".jpg")

                if label_name != image_name:
                    print(f"{label_name} != {image_name}")
                    continue

                pair_ls.append((label_info, image_info))

            for label_info, image_info in pair_ls:
                image_bytes = image_zip.open(image_info).read()
                label_bytes = label_zip.open(label_info).read()

                encoding_type = json.detect_encoding(label_bytes)
                label = json.loads(label_bytes.decode(encoding_type))

                old_bbox_ls = label.pop("Bbox")
                annotation = label.pop("Annotation")
                dataset = label.pop("Dataset")
                images = label.pop("Images")

                new_bbox_ls = list()
                for bbox in old_bbox_ls:
                    x_cord = sorted(set(bbox["x"]))
                    y_cord = sorted(set(bbox["y"]))

                    if len(x_cord) != 2:
                        # 무조건 x_min, y_min, x_max, y_max 값이 있어야지 bbox를 표시할 수 있는데 이건 하나 밖에 없어서 안됨
                        # 이 데이터는 특이하게 [123, 234, 123, 234]와 같이 표기되어 있는데 데이터 설명서에도 관련된 설명이 없음.
                        # 그래서 set을 한 뒤 sort를 하면 정상적으로 bbox가 맺히는걸 확인 했으나
                        # 그런 케이스에 들어가지 않는 포맷이라면 필터링 하기로 함.
                        print(f"""{bbox["x"]}\n{bbox["y"]}""")
                        print(f"{image_info.filename}는 bbox가 이상해서 패스 함")
                        continue

                    if len(y_cord) != 2:
                        print(f"""{bbox["x"]}\n{bbox["y"]}""")
                        print(f"{image_info.filename}는 패스 함")
                        continue

                    # [x_min, y_min, x_max, y_max]
                    try:
                        coordinate = (x_cord[0], y_cord[0], x_cord[1], y_cord[1])
                    except:
                        breakpoint()
                    meta = {"type": bbox["type"], "text_type": bbox["typeface"]}

                    new_bbox = {
                        "id": bbox["id"],
                        "text": bbox["data"],
                        "bbox": coordinate,
                        "meta": meta,
                    }
                    new_bbox_ls.append(new_bbox)

                meta_data = {}
                meta_data.update(annotation)
                meta_data.update(dataset)
                meta_data.update(images)

                label["objects"] = new_bbox_ls
                label["image"] = image_bytes
                label["meta_data"] = meta_data

                yield (idx_count, label)
                idx_count += 1
