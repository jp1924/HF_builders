# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""COCO"""

import json
import os
import random
from pathlib import Path
from tarfile import TarFile

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
    Version,
)
from natsort import natsorted
from tqdm import tqdm

_CITATION = """
@article{DBLP:journals/corr/LinMBHPRDZ14,
  author    = {Tsung{-}Yi Lin and
               Michael Maire and
               Serge J. Belongie and
               Lubomir D. Bourdev and
               Ross B. Girshick and
               James Hays and











               Pietro Perona and
               Deva Ramanan and
               Piotr Doll{\'{a}}r and
               C. Lawrence Zitnick},
  title     = {Microsoft {COCO:} Common Objects in Context},
  journal   = {CoRR},
  volume    = {abs/1405.0312},
  year      = {2014},
  url       = {http://arxiv.org/abs/1405.0312},
  eprinttype = {arXiv},
  eprint    = {1405.0312},
  timestamp = {Mon, 13 Aug 2018 16:48:13 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/LinMBHPRDZ14.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """
MS COCO is a large-scale object detection, segmentation, and captioning dataset.
COCO has several features: Object segmentation, Recognition in context, Superpixel stuff segmentation, 330K images (>200K labeled), 1.5 million object instances, 80 object categories, 91 stuff categories, 5 captions per image, 250,000 people with keypoints.
"""

_HOMEPAGE = "https://cocodataset.org/#home"

_LICENSE = "CC BY 4.0"


_IMAGES_URLS = {
    "train": "http://images.cocodataset.org/zips/train2014.zip",
    "validation": "http://images.cocodataset.org/zips/val2014.zip",
}

_KARPATHY_FILES_URL = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"

_SPLIT_MAP = {"train": "train2014", "validation": "val2014"}


DATASET_KEY = "261"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/{DATASET_KEY}.do"
_HOMEPAGE = (
    f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"
)

_VERSION = "1.0.0"
_DATANAME = "KoreanImageCaptioningDataset"


class KoreanImageCaptioningDataset(GeneratorBasedBuilder):
    VERSION = Version(_VERSION)

    BUILDER_CONFIGS = [
        BuilderConfig(
            name="default",
            version=VERSION,
            description="Same as 2014 but with all captions of one image gathered in a single example",
        ),
    ]

    DEFAULT_CONFIG_NAME = "default"
    VERSION = _VERSION

    def _info(self):
        _FEATURES = Features(
            {
                "id": Value("int32"),
                "image": Image(),
                "caption": Value("string"),
                "caption_ls": [Value("string")],
                "category": Value("string"),
                "filepath": Value("string"),
                "sentids": [Value("int32")],
                "filename": Value("string"),
                "split": Value("string"),
                "sentences_tokens": [[Value("string")]],
                "en_sentences_raw": [Value("string")],
                "sentences_sentid": [Value("int32")],
                "cocoid": Value("int32"),
            }
        )
        return DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
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

    def concat_json_part(self, unzip_dir: Path) -> None:
        part_glob = Path(unzip_dir).rglob("*.json.part*")

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

    def _split_generators(self, dl_manager):
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

            self.concat_json_part(unzip_dir)

        json_file_path = list(unzip_dir.rglob("*.json"))[0]

        annotation_file = os.path.join(dl_manager.download_and_extract(_KARPATHY_FILES_URL), "dataset_coco.json")
        image_folders = {k: Path(v) for k, v in dl_manager.download_and_extract(_IMAGES_URLS).items()}

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "annotation_file": annotation_file,
                    "ko_annotation_file": json_file_path,
                    "image_folders": image_folders,
                    "split_key": "train",
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "annotation_file": annotation_file,
                    "ko_annotation_file": json_file_path,
                    "image_folders": image_folders,
                    "split_key": "validation",
                },
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "annotation_file": annotation_file,
                    "ko_annotation_file": json_file_path,
                    "image_folders": image_folders,
                    "split_key": "test",
                },
            ),
        ]

    def _generate_examples(self, annotation_file, ko_annotation_file: Path, image_folders, split_key):
        random.seed(42)
        with open(annotation_file, "r", encoding="utf-8") as fi:
            annotations = json.load(fi)
            ko_annotation = json.loads(ko_annotation_file.read_text())
            ko_annotation = {x["id"]: x for x in ko_annotation}

            for idx, image_metadata in enumerate(annotations["images"]):
                if split_key == "train":
                    if image_metadata["split"] != "train" and image_metadata["split"] != "restval":
                        continue
                elif split_key == "validation":
                    if image_metadata["split"] != "val":
                        continue
                elif split_key == "test":
                    if image_metadata["split"] != "test":
                        continue

                if "val2014" in image_metadata["filename"]:
                    image_path = image_folders["validation"] / _SPLIT_MAP["validation"]
                else:
                    image_path = image_folders["train"] / _SPLIT_MAP["train"]

                image_path = image_path / image_metadata["filename"]

                record = {
                    "id": image_metadata["imgid"],
                    "image": Path(image_path.absolute()).read_bytes(),
                    "caption": random.choice(ko_annotation[image_metadata["cocoid"]]["caption_ko"]),
                    "caption_ls": ko_annotation[image_metadata["cocoid"]]["caption_ko"],
                    "category": None,
                    "filepath": image_metadata["filename"],
                    "sentids": image_metadata["sentids"],
                    "filename": image_metadata["filename"],
                    "split": image_metadata["split"],
                    "cocoid": image_metadata["cocoid"],
                    "sentences_tokens": [caption["tokens"] for caption in image_metadata["sentences"]],
                    "en_sentences_raw": [caption["raw"] for caption in image_metadata["sentences"]],
                    "sentences_sentid": [caption["sentid"] for caption in image_metadata["sentences"]],
                }

                yield idx, record
