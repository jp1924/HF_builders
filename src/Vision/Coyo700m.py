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


logging.basicConfig(filename="Coyo400m_download_fail.log", level=logging.INFO)
warnings.filterwarnings("error")

URLS = {
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


_HOMEPAGE = "https://huggingface.co/datasets/kakaobrain/coyo-700m"
_LICENSE = "The COYO dataset of Kakao Brain is licensed under CC-BY-4.0 License. The full license can be found in the LICENSE.cc-by-4.0 file. The dataset includes “Image URL” and “Text” collected from various sites by analyzing Common Crawl data, an open data web crawling project. The collected data (images and text) is subject to the license to which each content belongs."
_CITATION = """@misc{kakaobrain2022coyo-700m,
  title         = {COYO-700M: Image-Text Pair Dataset},
  author        = {Byeon, Minwoo and Park, Beomhee and Kim, Haecheon and Lee, Sungjun and Baek, Woonhyuk and Kim, Saehoon},
  year          = {2022},
  howpublished  = {\\url{https://github.com/kakaobrain/coyo-dataset}},
}"""
_DESCRIPTION = """COYO-700M is a large-scale dataset that contains 747M image-text pairs as well as many other meta-attributes to increase the usability to train various models. Our dataset follows a similar strategy to previous vision-and-language datasets, collecting many informative pairs of alt-text and its associated image in HTML documents. We expect COYO to be used to train popular large-scale foundation models complementary to other similar datasets."""
_VERSION = Version("1.0.0")


class Coyo400m(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [BuilderConfig(name="default", version=_VERSION)]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        features = Features(
            {
                "id": Value("int64"),
                "image": Image(),
                "caption": Value("string"),
                "caption_ls": [Value("string")],
                "category": Value("string"),
                "num_faces": Value("int32"),
                "nsfw_score": Value("float32"),
                "similarity": Value("float32"),
                "watermark_score": Value("float32"),
                "aesthetic_score": Value("float32"),
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
            sample_id_ls = example["id"]
            sample_id_ls = sample_id_ls = sample_id_ls if isinstance(sample_id_ls, list) else [sample_id_ls]

            url_ls = example["url"]
            url_ls = url_ls = url_ls if isinstance(url_ls, list) else [url_ls]

            text_ls = example["text"]
            text_ls = text_ls = text_ls if isinstance(text_ls, list) else [text_ls]

            num_faces_ls = example["num_faces"]
            num_faces_ls = num_faces_ls = num_faces_ls if isinstance(num_faces_ls, list) else [num_faces_ls]

            nsfw_ls = example["nsfw_score_gantman"]
            nsfw_ls = nsfw_ls = nsfw_ls if isinstance(nsfw_ls, list) else [nsfw_ls]

            similarity_ls = example["clip_similarity_vitl14"]
            similarity_ls = similarity_ls = similarity_ls if isinstance(similarity_ls, list) else [similarity_ls]

            watermark_ls = example["watermark_score"]
            watermark_ls = watermark_ls = watermark_ls if isinstance(watermark_ls, list) else [watermark_ls]

            aesthetic_ls = example["aesthetic_score_laion_v2"]
            aesthetic_ls = aesthetic_ls = aesthetic_ls if isinstance(aesthetic_ls, list) else [aesthetic_ls]

            data = {
                "id": [],
                "image": [],
                "caption": [],
                "caption_ls": [],
                "category": [],
                "num_faces": [],
                "nsfw_score": [],
                "similarity": [],
                "watermark_score": [],
                "aesthetic_score": [],
            }
            iter_zip = zip(
                sample_id_ls, url_ls, text_ls, num_faces_ls, nsfw_ls, similarity_ls, watermark_ls, aesthetic_ls
            )
            for sample_id, url, text, num_faces, nsfw, similarity, watermark, aesthetic in iter_zip:
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

                image = PIL_Image.open(BytesIO(response.content))

                data["id"].append(sample_id)
                data["image"].append(image)
                data["caption"].append(text)
                data["caption_ls"].append([text])
                data["category"].append(None)
                data["num_faces"].append(num_faces)
                data["nsfw_score"].append(nsfw)
                data["similarity"].append(similarity)
                data["watermark_score"].append(watermark)
                data["aesthetic_score"].append(aesthetic)

            return data

        idx_ = 0
        for idx, parquet_path in enumerate(filepath.values()):
            dataset = load_dataset("parquet", data_files=[parquet_path], split="train")

            parquet_path = Path(parquet_path)
            cache_file_path = parquet_path.parent.joinpath("coyo700m_cache_file", f"coyo700m-{idx}_cache_file.arrow")

            if not cache_file_path.parent.exists():
                print(f"mkdir coyo700m_cache_file at {cache_file_path.parent}")
                cache_file_path.parent.mkdir()

            dataset = dataset.map(
                download_img,
                num_proc=40,
                batched=True,
                batch_size=10,
                load_from_cache_file=True,
                desc=f"Coyo400m-{idx}",
                with_rank=True,
                cache_file_name=str(cache_file_path),
                remove_columns=dataset.column_names,
            )
            for data in dataset:
                yield (idx_, data)
                idx_ += 1
