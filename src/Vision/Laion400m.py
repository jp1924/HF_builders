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


logging.basicConfig(filename="Laion400m_download_fail.log", level=logging.INFO)
warnings.filterwarnings("error")

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

_HOMEPAGE = "https://laion.ai/blog/laion-400-open-dataset/"
_LICENSE = "We distribute the metadata dataset (the parquet files) under the most open Creative Common CC-BY 4.0 license, which poses no particular restriction. The images are under their copyright."
_CITATION = """Schuhmann, C., Vencu, R., Beaumont, R., Kaczmarczyk, R., Mullis, C., Katta, A., Coombes, T., Jitsev, J., & Komatsuzaki, A. (2021). LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs. arXiv. https://doi.org/10.48550/arXiv.2111.02114"""
_DESCRIPTION = """Multi-modal language-vision models trained on hundreds of millions of image-text pairs (e.g. CLIP, DALL-E) gained a recent surge, showing remarkable capability to perform zero- or few-shot learning and transfer even in absence of per-sample labels on target image data. Despite this trend, to date there has been no publicly available datasets of sufficient scale for training such models from scratch. To address this issue, in a community effort we build and release for public LAION-400M, a dataset with CLIP-filtered 400 million image-text pairs, their CLIP embeddings and kNN indices that allow efficient similarity search."""
_VERSION = Version("1.0.0")


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
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            version=_VERSION,
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

            text_ls = example["TEXT"]
            text_ls = text_ls = text_ls if isinstance(text_ls, list) else [text_ls]

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
            iter_zip = zip(sample_id_ls, url_ls, text_ls, license_ls, nsfw_ls, similarity_ls)
            for sample_id, url, text, license, nsfw, similarity in iter_zip:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code != 200:
                        logging.info(f"{sample_id} is skip")
                        continue

                    img_bytes = BytesIO(response.content)
                    # 종종 byte가 다운 받아져도 열리지 않는 깨진 이미지가 다운 받아지는 경우가 있음
                    # 그리고 warning이 뜨면 error가 나도록 만들어 놨는데 정상 동작하는지 테스트는 안했음.
                    img = PIL_Image.open(img_bytes)
                    img.load()
                    img.verify()

                    if img.format not in ["JPEG", "PNG", "WebP"]:
                        raise ValueError()

                    # 10 이하의 이미지들은 대부분 다 필터링 하도록 만듬.
                    if img.width < 10 or img.height < 10:
                        raise ValueError()

                    if not (text or sample_id):
                        raise ValueError()

                except:
                    logging.info(f"{sample_id} is skip")
                    continue

                image = PIL_Image.open(BytesIO(response.content)).convert("RGB")

                data["id"].append(sample_id)
                data["image"].append(image)
                data["caption"].append(text)
                data["caption_ls"].append([text])
                data["category"].append(None)
                data["license"].append(license)
                data["nsfw"].append(nsfw)
                data["similarity"].append(similarity)

            return data

        idx_ = 0
        for idx, parquet_path in enumerate(filepath.values()):
            dataset = load_dataset("parquet", data_files=[parquet_path], split="train")

            parquet_path = Path(parquet_path)
            cache_file_path = parquet_path.parent.joinpath("Laion400m_cache_file", f"Laion400m-{idx}_cache_file.arrow")

            if not cache_file_path.parent.exists():
                print(f"mkdir Laion400m_cache_file at {cache_file_path.parent}")
                cache_file_path.parent.mkdir()

            dataset = dataset.map(
                download_img,
                num_proc=30,
                batched=True,
                batch_size=10,
                load_from_cache_file=True,
                desc=f"Laion400m-{idx}",
                with_rank=True,
                cache_file_name=str(cache_file_path),
                remove_columns=dataset.column_names,
            )
            for data in dataset:
                yield (idx_, data)
                idx_ += 1
