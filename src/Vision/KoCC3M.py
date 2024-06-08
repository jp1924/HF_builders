import logging
import warnings
from io import BytesIO

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
    concatenate_datasets,
    load_dataset,
)
from PIL import Image as PIL_Image
from setproctitle import setproctitle


logging.basicConfig(filename="download_fail.log", level=logging.INFO)

_DESCRIPTION = """CC12M of flax-community/conceptual-captions-12 translated from English to Korean."""

TRAIN_URLs = {
    "157cf2686bec271f": "https://huggingface.co/datasets/QuoQA-NLP/KoCC3M/resolve/main/data/train-00000-of-00002-cc8e11261b9f26e1.parquet?download=true",
    "27c90b30911ef7aa": "https://huggingface.co/datasets/QuoQA-NLP/KoCC3M/resolve/main/data/train-00001-of-00002-3d1333b77c91c8c1.parquet?download=true",
}

VALID_URLs = {
    "27c90b30911ef7aa": "https://huggingface.co/datasets/QuoQA-NLP/KoCC3M/resolve/main/data/validation-00000-of-00001-168f14d7fd7256ba.parquet?download=true",
}

setproctitle("KoCC3M_builder")


class WarningAsException(Exception):
    pass


def warning_to_exception(message, category, filename, lineno, file=None, line=None):
    raise WarningAsException(message)


class KoCC3M(GeneratorBasedBuilder):
    VERSION = Version("1.0.0")

    def _info(self):
        features = Features(
            {
                "id": Value("int32"),
                "image": Image(),
                "caption": Value("string"),
                "caption_ls": [Value("string")],
                "category": Value("string"),
                "en_caption": Value("string"),
            }
        )

        return DatasetInfo(
            features=features,
            supervised_keys=None,
            citation=None,
        )

    def _split_generators(self, dl_manager):
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "filepath": dl_manager.download_and_extract(TRAIN_URLs),
                    "split": "train",
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "filepath": dl_manager.download_and_extract(VALID_URLs),
                    "split": "valid",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        warnings.showwarning = warning_to_exception

        def download_img(example):
            url_ls = example["image_url"]
            url_ls = url_ls = url_ls if isinstance(url_ls, list) else [url_ls]

            english_caption_ls = example["english_caption"]
            english_caption_ls = english_caption_ls = (
                english_caption_ls if isinstance(english_caption_ls, list) else [english_caption_ls]
            )

            korean_caption_ls = example["korean_caption"]
            korean_caption_ls = korean_caption_ls = (
                korean_caption_ls if isinstance(korean_caption_ls, list) else [korean_caption_ls]
            )

            data = {
                "image": [],
                "caption": [],
                "caption_ls": [],
                "category": [],
                "en_caption": [],
            }
            for url, korean_caption, english_caption in zip(url_ls, korean_caption_ls, english_caption_ls):
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code != 200:
                        logging.info(f"{url} is skip")
                        continue

                    response.raise_for_status()
                    img_bytes = BytesIO(response.content)
                    PIL_Image.open(img_bytes)
                except WarningAsException as e:
                    logging.info(f"{url} is warning and skip")
                    continue
                except:
                    logging.info(f"{url} is skip")
                    continue
                data["image"].append(PIL_Image.open(BytesIO(response.content)))
                data["caption"].append(korean_caption)
                data["caption_ls"].append([korean_caption])
                data["category"].append(None)
                data["en_caption"].append(english_caption)

            return data

        idx = 1
        for parquet_path in filepath.values():
            part_dataset = load_dataset("parquet", data_files=parquet_path, split="train")
            part_dataset = part_dataset.map(
                download_img,
                num_proc=40,
                batched=True,
                batch_size=10,
                remove_columns=part_dataset.column_names,
            )
            for row in part_dataset:
                row["id"] = idx

                yield (idx, row)
                idx += 1
