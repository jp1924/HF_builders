import logging
from io import BytesIO

import PIL
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
    load_dataset,
)
from setproctitle import setproctitle


logging.basicConfig(filename="download_fail.log", level=logging.INFO)

_DESCRIPTION = """CC12M of flax-community/conceptual-captions-12 translated from English to Korean."""

_URLs = {
    "157cf2686bec271f": "https://huggingface.co/datasets/QuoQA-NLP/KoCC12M/resolve/main/data/train-00000-of-00009-157cf2686bec271f.parquet?download=true",
    "27c90b30911ef7aa": "https://huggingface.co/datasets/QuoQA-NLP/KoCC12M/resolve/main/data/train-00001-of-00009-27c90b30911ef7aa.parquet?download=true",
    "827d4088a57b200a": "https://huggingface.co/datasets/QuoQA-NLP/KoCC12M/resolve/main/data/train-00002-of-00009-827d4088a57b200a.parquet?download=true",
    "52d7cd2b0f7eadf4": "https://huggingface.co/datasets/QuoQA-NLP/KoCC12M/resolve/main/data/train-00003-of-00009-52d7cd2b0f7eadf4.parquet?download=true",
    "12f9b17a7d230ec9": "https://huggingface.co/datasets/QuoQA-NLP/KoCC12M/resolve/main/data/train-00004-of-00009-12f9b17a7d230ec9.parquet?download=true",
    "561e693a58593192": "https://huggingface.co/datasets/QuoQA-NLP/KoCC12M/resolve/main/data/train-00005-of-00009-561e693a58593192.parquet?download=true",
    "020e6498d3fa2a1a": "https://huggingface.co/datasets/QuoQA-NLP/KoCC12M/resolve/main/data/train-00006-of-00009-020e6498d3fa2a1a.parquet?download=true",
    "78c2750fc6ff13f1": "https://huggingface.co/datasets/QuoQA-NLP/KoCC12M/resolve/main/data/train-00007-of-00009-78c2750fc6ff13f1.parquet?download=true",
    "5aa45daa7c31925c": "https://huggingface.co/datasets/QuoQA-NLP/KoCC12M/resolve/main/data/train-00008-of-00009-5aa45daa7c31925c.parquet?download=true",
}


setproctitle("KoCC3M_builder")


class KoCC12M(GeneratorBasedBuilder):
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
            description=_DESCRIPTION,
        )

    def _split_generators(self, dl_manager):
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "filepath": dl_manager.download_and_extract(_URLs),
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        def download_img(example):
            url_ls = example["url"]
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
                    PIL.Image.open(img_bytes)
                except:
                    logging.info(f"{url} is skip")
                    continue
                data["image"].append(img_bytes.read())
                data["caption"].append(korean_caption)
                data["caption_ls"].append([korean_caption])
                data["category"].append(None)
                data["en_caption"].append(english_caption)

            return data

        idx = 1
        for parquet_path in filepath.values():
            part_dataset = load_dataset("parquet", data_files=parquet_path, split=split)
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
