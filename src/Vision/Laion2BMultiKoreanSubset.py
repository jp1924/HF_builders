import logging
import warnings
from io import BytesIO

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
    load_dataset,
)
from PIL import Image as PIL_Image
from setproctitle import setproctitle


logging.basicConfig(filename="Laion2BMultiKoreanSubset_download_fail.log", level=logging.INFO)


_URLs = {
    "157cf2686bec271f": "https://huggingface.co/datasets/Bingsu/laion2B-multi-korean-subset/resolve/main/data/train-00000-of-00006-5fe89370890da24f.parquet?download=true",
    "27c90b30911ef7aa": "https://huggingface.co/datasets/Bingsu/laion2B-multi-korean-subset/resolve/main/data/train-00001-of-00006-aab9b1bf52a2df3f.parquet?download=true",
    "827d4088a57b200a": "https://huggingface.co/datasets/Bingsu/laion2B-multi-korean-subset/resolve/main/data/train-00002-of-00006-1c4b4049541a99d4.parquet?download=true",
    "52d7cd2b0f7eadf4": "https://huggingface.co/datasets/Bingsu/laion2B-multi-korean-subset/resolve/main/data/train-00003-of-00006-9f022966b6bb9cf8.parquet?download=true",
    "12f9b17a7d230ec9": "https://huggingface.co/datasets/Bingsu/laion2B-multi-korean-subset/resolve/main/data/train-00004-of-00006-829cbdc252d37dd4.parquet?download=true",
    "561e693a58593192": "https://huggingface.co/datasets/Bingsu/laion2B-multi-korean-subset/resolve/main/data/train-00005-of-00006-5435e409ba6a048d.parquet?download=true",
}


setproctitle("Laion2BMultiKoreanSubset_builder")


class WarningAsException(Exception):
    pass


def warning_to_exception(message, category, filename, lineno, file=None, line=None):
    raise WarningAsException(message)


class Laion2BMultiKoreanSubset(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="default",
            version=Version("1.0.0"),
            description="a subset data of laion/laion2B-multi, including only korean",
        ),
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        features = Features(
            {
                "id": Value("int32"),
                "image": Image(),
                "caption": Value("string"),
                "caption_ls": [Value("string")],
                "category": Value("string"),
                "language": Value("string"),
                "nsfw": Value("string"),
                "license": Value("string"),
                "similarity": Value("float32"),
            }
        )

        return DatasetInfo(
            features=features,
            supervised_keys=None,
            citation=None,
            description="a subset data of laion/laion2B-multi, including only korean",
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
        warnings.showwarning = warning_to_exception

        def download_img(example):
            url_ls = example["URL"]
            url_ls = url_ls = url_ls if isinstance(url_ls, list) else [url_ls]

            sample_id_ls = example["SAMPLE_ID"]
            sample_id_ls = sample_id_ls = sample_id_ls if isinstance(sample_id_ls, list) else [sample_id_ls]

            license_ls = example["LICENSE"]
            license_ls = license_ls = license_ls if isinstance(license_ls, list) else [license_ls]

            nsfw_ls = example["NSFW"]
            nsfw_ls = nsfw_ls = nsfw_ls if isinstance(nsfw_ls, list) else [nsfw_ls]

            language_ls = example["LANGUAGE"]
            language_ls = language_ls = language_ls if isinstance(language_ls, list) else [language_ls]

            similarity_ls = example["similarity"]
            similarity_ls = similarity_ls = similarity_ls if isinstance(similarity_ls, list) else [similarity_ls]

            korean_caption_ls = example["TEXT"]
            korean_caption_ls = korean_caption_ls = (
                korean_caption_ls if isinstance(korean_caption_ls, list) else [korean_caption_ls]
            )

            data = {
                "id": [],
                "image": [],
                "caption": [],
                "caption_ls": [],
                "category": [],
                "license": [],
                "nsfw": [],
                "language": [],
                "similarity": [],
            }
            for url, korean_caption, license, nsfw, language, similarity, sample_id in zip(
                url_ls,
                korean_caption_ls,
                license_ls,
                nsfw_ls,
                language_ls,
                similarity_ls,
                sample_id_ls,
            ):
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code != 200:
                        logging.info(f"{url} is skip")
                        continue

                    response.raise_for_status()
                    img_bytes = BytesIO(response.content)
                    PIL_Image.open(img_bytes).load()
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
                data["license"].append(license)
                data["nsfw"].append(nsfw)
                data["language"].append(language)
                data["similarity"].append(similarity)
                data["id"].append(sample_id)

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
                # pyarrow.lib.ArrowInvalid: Integer value 3737204013042 not in range: -2147483648 to 2147483647
                # 같은 애러가 발생해서 id를 재 정의 함.
                row["id"] = idx
                yield (idx, row)
                idx += 1
