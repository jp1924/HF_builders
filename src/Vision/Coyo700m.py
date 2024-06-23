import logging
import warnings
from io import BytesIO
from pathlib import Path

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


URL = {
    "part-00000-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00000-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00001-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00001-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00002-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00002-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00003-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00003-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00004-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00004-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00005-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00005-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00006-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00006-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00007-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00007-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00008-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00008-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00009-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00009-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00010-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00010-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00011-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00011-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00012-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00012-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00013-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00013-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00014-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00014-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00015-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00015-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00016-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00016-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00017-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00017-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00018-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00018-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00019-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00019-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00020-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00020-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00021-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00021-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00022-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00022-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00023-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00023-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00024-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00024-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00025-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00025-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00026-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00026-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00027-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00027-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00028-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00028-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00029-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00029-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00030-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00030-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00031-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00031-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00032-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00032-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00033-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00033-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00034-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00034-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00035-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00035-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00036-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00036-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00037-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00037-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00038-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00038-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00039-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00039-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00040-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00040-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00041-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00041-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00042-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00042-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00043-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00043-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00044-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00044-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00045-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00045-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00046-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00046-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00047-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00047-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
    "part-00048-17da4908-939c-46e5-91d0-15f256041956-c000": "https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00048-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet?download=true",
}


PATH = "https://huggingface.co/datasets/kakaobrain/coyo-700m"
_VERSION = Version("1.0.0")


class WarningAsException(Exception):
    pass


def warning_to_exception(message, category, filename, lineno, file=None, line=None):
    raise WarningAsException(message)


class Coyo400m(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [BuilderConfig(name="default", version=_VERSION)]
    DEFAULT_CONFIG_NAME = "default"
    VERSION = _VERSION

    def _info(self):
        features = Features(
            {
                "id": Value("int64"),
                "image": Image(),
                "caption": Value("string"),
                "caption_ls": [Value("string")],
                "category": Value("string"),
                "license": Value("string"),
                "nsfw": Value("string"),
                "similarity": Value("float"),
            }
        )

        return DatasetInfo(
            features=features,
            supervised_keys=None,
            citation=None,
        )

    def _split_generators(self, dl_manager):
        download_file = dl_manager.download_and_extract(URLS)

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "filepath": download_file,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        def download_img(example, rank):
            sample_id_ls = example["SAMPLE_ID"]
            sample_id_ls = sample_id_ls = sample_id_ls if isinstance(sample_id_ls, list) else [sample_id_ls]

            url_ls = example["URL"]
            url_ls = url_ls = url_ls if isinstance(url_ls, list) else [url_ls]

            korean_caption_ls = example["TEXT"]
            korean_caption_ls = korean_caption_ls = (
                korean_caption_ls if isinstance(korean_caption_ls, list) else [korean_caption_ls]
            )

            license_ls = example["LICENSE"]
            license_ls = license_ls = license_ls if isinstance(license_ls, list) else [license_ls]

            nsfw_ls = example["NSFW"]
            nsfw_ls = nsfw_ls = nsfw_ls if isinstance(nsfw_ls, list) else [nsfw_ls]

            similarity_ls = example["similarity"]
            similarity_ls = similarity_ls = similarity_ls if isinstance(similarity_ls, list) else [similarity_ls]

            data = {
                "id": [],
                "image": [],
                "caption": [],
                "caption_ls": [],
                "category": [],
                "license": [],
                "nsfw": [],
                "similarity": [],
            }
            iter_zip = zip(sample_id_ls, url_ls, korean_caption_ls, license_ls, nsfw_ls, similarity_ls)
            for sample_id, url, korean_caption, license, nsfw, similarity in iter_zip:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code != 200:
                        logging.info(f"{url} is skip")
                        continue

                    img_bytes = BytesIO(response.content)
                    # 종종 byte가 다운 받아져도 열리지 않는 깨진 이미지가 다운 받아지는 경우가 있음
                    # 그리고 warning이 뜨면 error가 나도록 만들어 놨는데 정상 동작하는지 테스트는 안했음.
                    PIL_Image.open(img_bytes).load()
                except WarningAsException as e:
                    logging.info(f"{url} is warning and skip")
                    continue
                except:
                    logging.info(f"{url} is skip")
                    continue

                image = PIL_Image.open(BytesIO(response.content))

                data["id"].append(sample_id)
                data["image"].append(image)
                data["caption"].append(korean_caption)
                data["caption_ls"].append([korean_caption])
                data["category"].append(None)
                data["license"].append(license)
                data["nsfw"].append(nsfw)
                data["similarity"].append(similarity)

            return data

        idx_ = 0
        warnings.showwarning = warning_to_exception
        for idx, parquet_path in enumerate(filepath.values()):
            dataset = load_dataset("parquet", data_files=[parquet_path], split="train")

            parquet_path = Path(parquet_path)
            cache_file_path = parquet_path.parent.joinpath(f"Laion400m-{idx}_cache_file.arrow")
            dataset = dataset.map(
                download_img,
                num_proc=2,
                batched=True,
                batch_size=1000,
                load_from_cache_file=True,
                desc=f"Laion400m-{idx}",
                with_rank=True,
                cache_file_name=str(cache_file_path),
                remove_columns=dataset.column_names,
            )
            for data in dataset:
                yield (idx_, data)
                idx_ += 1
