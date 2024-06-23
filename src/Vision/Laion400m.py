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


URLS = {
    "part-00000-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00000-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00001-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00001-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00002-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00002-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00003-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00003-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00004-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00004-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00005-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00005-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00006-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00006-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00007-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00007-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00008-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00008-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00009-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00009-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00010-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00010-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00011-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00011-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00012-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00012-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00013-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00013-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00014-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00014-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00015-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00015-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00016-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00016-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00017-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00017-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00018-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00018-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00019-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00019-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00020-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00020-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00021-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00021-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00022-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00022-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00023-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00023-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00024-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00024-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00025-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00025-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00026-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00026-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00027-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00027-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00028-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00028-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00029-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00029-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00030-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00030-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
    "part-00031-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000": "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00031-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
}

_VERSION = Version("1.0.0")


class WarningAsException(Exception):
    pass


def warning_to_exception(message, category, filename, lineno, file=None, line=None):
    raise WarningAsException(message)


class Laion400m(GeneratorBasedBuilder):
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
