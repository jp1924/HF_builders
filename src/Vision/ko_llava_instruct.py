# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
from tarfile import TarFile
from typing import List
from zipfile import ZipExtFile, ZipFile

import requests
from datasets import (
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


URLS = {
    "label": "https://huggingface.co/datasets/tabtoyou/KoLLaVA-v1.5-Instruct-581k/resolve/main/kollava_v1_5_mix581k.json?download=true",
    "VisualGenome_1": "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
    "VisualGenome_2": "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
    "GQA_image": "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
    "coco_image": "http://images.cocodataset.org/zips/train2017.zip",
}

DATASET_KEY = "71357"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/{DATASET_KEY}.do"
DATASET_SIZE = 11.55


class KoLLaVAInsturct(GeneratorBasedBuilder):
    VERSION = Version("1.1.0")

    def _info(self):
        return DatasetInfo(
            features=Features(
                {
                    "image": Image(),
                    "conversations": [{"from": Value("string"), "value": Value("string")}],
                    "id": Value("string"),
                }
            ),
            supervised_keys=None,
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

    def unzip_data(self, tar_file: Path, unzip_dir: Path) -> list:
        with TarFile(tar_file, "r") as mytar:
            mytar.extractall(unzip_dir)
            os.remove(tar_file)

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

        return list(unzip_dir.rglob("*.zip*"))

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(URLS)

        data_name = "Outside_Knowledge_based_Multimodal_QA_Data"
        cache_dir = Path(dl_manager.download_config.cache_dir)
        unzip_dir = cache_dir.joinpath(data_name)

        if not unzip_dir.exists():
            tar_file = cache_dir.joinpath(f"{data_name}.tar")
            self.aihub_downloader(tar_file)
            # 압축이 덜 출렸을 때를 고려해야 함.
            zip_file_path = self.unzip_data(tar_file, unzip_dir)
        else:
            zip_file_path = list(unzip_dir.rglob("*.zip"))

        source_ls = [ZipFile(x) for x in zip_file_path if "원천데이터" in str(x)]

        ekvqa_dict = dict()
        for zip_file in source_ls:
            for file in zip_file.filelist:
                ekvqa_dict[f"ekvqa{file.filename}"] = zip_file.open(file.filename)

        label = downloaded_files.pop("label")
        label = json.loads(Path(label).read_text())

        change_dict = {
            "VisualGenome_1": "vg",
            "VisualGenome_2": "vg",
            "GQA_image": "gqa",
            "coco_image": "coco",
        }
        image_dict = dict()
        total_len = 0
        for k, v in downloaded_files.items():
            prefix = change_dict[k]
            img_file = [x for x in Path(v).rglob("*") if not x.is_dir()]
            path_pair_dict = {str(x).replace(str(v), prefix): x for x in img_file}
            image_dict.update(path_pair_dict)
            total_len += len(img_file)

        image_dict.update(ekvqa_dict)

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "label_ls": label,
                    "image_dict": image_dict,
                },
            ),
        ]

    def _generate_examples(self, label_ls, image_dict):
        for idx, x in enumerate(label_ls):
            if x["image"] not in image_dict:
                continue

            image = image_dict[x["image"]]
            if isinstance(image, ZipExtFile):
                image = image.read()
            else:
                image = image.read_bytes()
            x["image"] = image

            yield (idx, x)
