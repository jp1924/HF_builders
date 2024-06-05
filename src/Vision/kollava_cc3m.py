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
    "image": "https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/resolve/main/images.zip?download=true",
    "label": "https://huggingface.co/datasets/tabtoyou/KoLLaVA-CC3M-Pretrain-595K/resolve/main/ko_chat.json?download=true",
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
                    "id": Value("string"),
                    "image": Image(),
                    "conversations": [{"role": Value("string"), "content": Value("string")}],
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(URLS)

        image = downloaded_files.pop("image")
        image_file_ls = [x for x in Path(image).rglob("*") if x.is_file()]
        image_file_table = {x.name: x for x in image_file_ls}

        label = downloaded_files.pop("label")
        label = json.loads(Path(label).read_text())

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "label_ls": label,
                    "image_dict": image_file_table,
                },
            ),
        ]

    def _generate_examples(self, label_ls, image_dict):
        for idx, x in enumerate(label_ls):
            if x["image"] not in image_dict:
                continue

            new_conversations_ls = list()
            for y in x["conversations"]:
                role = "user" if y["from"] == "human" else "assistant"
                new_conversations_ls.append({"role": role, "content": y["value"].replace("<image>", "").strip()})

            x["image"] = image_dict[x["image"]].read_bytes()
            x["conversations"] = new_conversations_ls

            yield (idx, x)
