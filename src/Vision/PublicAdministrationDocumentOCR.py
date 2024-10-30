# -*- coding: utf-8 -*-
import io
import json
import os
from pathlib import Path
from tarfile import TarFile
from typing import Generator, List
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
from PIL import Image as PIL_Image
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
_HOMEPAGE = (
    f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"
)

_VERSION = "1.2.0"
_DATANAME = "PublicAdministrationDocumentOCR"
DATASET_SIZE = 384.89


class PublicAdministrationDocumentOCR(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(name="chat", version=_VERSION),
        BuilderConfig(name="ocr", version=_VERSION),
    ]
    DEFAULT_CONFIG_NAME = "chat"

    def _info(self) -> DatasetInfo:
        if self.config_id == "chat":
            features = Features(
                {
                    "id": Value("int32"),
                    "image": Image(),
                    "conversations": [
                        {
                            "role": Value("string"),
                            "content": Value("string"),
                        },
                    ],
                }
            )
        elif self.config_id == "ocr":
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

    def _generate_examples(self, filepath: List[Path], split: str) -> Generator:
        if self.config_id == "chat":
            return self._chat_generate_examples(filepath, split)
        if self.config_id == "ocr":
            return self._ocr_generate_examples(filepath, split)

    def _ocr_generate_examples(self, filepath: List[Path], split: str) -> Generator:
        yield

    def _chat_generate_examples(self, filepath: List[Path], split: str) -> Generator:
        def group_bboxes_by_line(bbox_ls, y_threshold=10):
            # ["top-x", "top-y", "bottom-x", "bottom-y"]
            # wordbox: [2039, 3066, 2072, 3101]으로 되어 있음.
            lines = []
            current_line = [bbox_ls[0]]

            for bbox in bbox_ls[1:]:
                if bbox["bbox"][1] - current_line[-1]["bbox"][1] <= y_threshold:
                    current_line.append(bbox)
                else:
                    lines.append(current_line)
                    current_line = [bbox]

            lines.append(current_line)
            return lines

        def create_line_bbox(line):
            x_min = min(bbox["bbox"][0] for bbox in line)
            y_min = min(bbox["bbox"][1] for bbox in line)
            x_max = max(bbox["bbox"][2] for bbox in line)
            y_max = max(bbox["bbox"][3] for bbox in line)

            line = sorted(line, key=lambda bbox: bbox["bbox"][0])

            return {
                "bbox": [x_min, y_min, x_max, y_max],
                "text": " ".join(bbox["text"] for bbox in line),
            }

        source_zip_ls = [ZipFile(x) for x in filepath if "[원천]" in str(x)]
        source_zip_ls = natsorted(source_zip_ls, key=lambda x: x.filename)

        label_zip_ls = [ZipFile(x) for x in filepath if "[라벨]" in str(x)]
        label_zip_ls = natsorted(label_zip_ls, key=lambda x: x.filename)

        label_image_match_dict = dict()
        for label_zip in label_zip_ls:
            label_info_ls = [file for file in label_zip.filelist if ".json" in file.filename]
            label_info_dict = {Path(file.filename).stem: label_zip.open(file).read() for file in label_info_ls}
            label_image_match_dict.update(label_info_dict)

        _idx = 0
        for source_zip in source_zip_ls:
            source_info_ls = [file for file in source_zip.filelist if ".jpg" in file.filename]
            source_info_ls = natsorted(source_info_ls, key=lambda x: x.filename)

            for source_info in source_info_ls:
                # check image $ label file is matching
                filename = Path(source_info.filename).stem
                if filename not in label_image_match_dict:
                    print(f"{filename}가 없어서 해당 파일은 패스 함.")
                    continue

                # load image & label
                try:
                    label = label_image_match_dict[filename].decode("utf-8")
                    label = json.loads(label)
                    image = source_zip.open(source_info).read()
                    image = PIL_Image.open(io.BytesIO(image))
                except BaseException as e:
                    print(f"{e} 애러가 발생해서 해당 이미지는 스킵함.")
                    continue

                # fix label annotations
                new_annotation_ls = list()
                for annotation in label["annotations"]:
                    annotation = {k.split(".")[-1]: v for k, v in annotation.items()}

                    # [598, 1623, 323, 63] 같이 되어 있음.
                    top_x, top_y, width, height = annotation["bbox"]
                    new_bbox = [top_x, top_y, top_x + width, top_y + height]
                    annotation["bbox"] = new_bbox

                    new_annotation_ls.append(annotation)

                # gathering obj bbox
                lines = group_bboxes_by_line(new_annotation_ls)
                lines = [create_line_bbox(line) for line in lines]

                # cropping & formatting
                for line in lines:
                    try:
                        cropped_img = image.crop(line["bbox"])
                        rectangle_image = io.BytesIO()
                        cropped_img.save(rectangle_image, format="JPEG")
                    except BaseException as e:
                        print(line["bbox"])
                        continue

                    conversations = [
                        {"role": "user", "content": json.dumps([{"type": "image"}])},
                        {
                            "role": "assistant",
                            "content": json.dumps([{"type": "text", "text": line["text"]}], ensure_ascii=False),
                        },
                    ]
                    data = {
                        "id": _idx,
                        "image": rectangle_image.getvalue(),
                        "conversations": conversations,
                    }

                    yield (_idx, data)
                    _idx += 1
