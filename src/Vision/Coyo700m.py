import logging
from io import BytesIO
from pathlib import Path

import pyarrow as pa
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
from img2dataset.downloader import download_image_with_retry
from img2dataset.resizer import Resizer
from PIL import Image as PIL_Image


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


class Coyo700m(GeneratorBasedBuilder):
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
                "width": Value("int32"),
                "height": Value("int32"),
                "image_phash": Value("string"),
                "text_length": Value("int32"),
                "word_count": Value("int32"),
                "num_tokens_bert": Value("int32"),
                "num_tokens_gpt": Value("int32"),
                "num_faces": Value("int32"),
                "clip_similarity_vitb32": Value("float32"),
                "clip_similarity_vitl14": Value("float32"),
                "nsfw_score_opennsfw2": Value("float32"),
                "nsfw_score_gantman": Value("float32"),
                "watermark_score": Value("float32"),
                "aesthetic_score_laion_v2": Value("float32"),
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
        def image_downloader(example):
            breakpoint()
            (
                sample_id_ls,
                url_ls,
                text_ls,
                width_ls,
                height_ls,
                image_phash_ls,
                text_length_ls,
                word_count_ls,
                num_tokens_bert_ls,
                num_tokens_gpt_ls,
                num_faces_ls,
                clip_similarity_vitb32_ls,
                clip_similarity_vitl14_ls,
                nsfw_score_opennsfw2_ls,
                nsfw_score_gantman_ls,
                watermark_score_ls,
                aesthetic_score_laion_v2_ls,
            ) = (
                example["id"],
                example["url"],
                example["text"],
                example["width"],
                example["height"],
                example["image_phash"],
                example["text_length"],
                example["word_count"],
                example["num_tokens_bert"],
                example["num_tokens_gpt"],
                example["num_faces"],
                example["clip_similarity_vitb32"],
                example["clip_similarity_vitl14"],
                example["nsfw_score_opennsfw2"],
                example["nsfw_score_gantman"],
                example["watermark_score"],
                example["aesthetic_score_laion_v2"],
            )
            finish_data_ls = list()
            coyo_zip = zip(
                sample_id_ls,
                url_ls,
                text_ls,
                width_ls,
                height_ls,
                image_phash_ls,
                text_length_ls,
                word_count_ls,
                num_tokens_bert_ls,
                num_tokens_gpt_ls,
                num_faces_ls,
                clip_similarity_vitb32_ls,
                clip_similarity_vitl14_ls,
                nsfw_score_opennsfw2_ls,
                nsfw_score_gantman_ls,
                watermark_score_ls,
                aesthetic_score_laion_v2_ls,
            )
            for (
                sample_id,
                url,
                text,
                width,
                height,
                image_phash,
                text_length,
                word_count,
                num_tokens_bert,
                num_tokens_gpt,
                num_faces,
                clip_similarity_vitb32,
                clip_similarity_vitl14,
                nsfw_score_opennsfw2,
                nsfw_score_gantman,
                watermark_score,
                aesthetic_score_laion_v2,
            ) in coyo_zip:
                idx, io_stream, err = download_image_with_retry(
                    (sample_id, url),
                    timeout=5,
                    retries=2,
                    user_agent_token=None,
                    disallowed_header_directives=False,
                )
                if err:
                    continue
                img_bytes, _, _, orig_height, orig_width, err = resizer(io_stream)
                if height != orig_height or width != orig_width:
                    continue
                elif err:
                    continue
                pil_image = PIL_Image.open(BytesIO(img_bytes))
                pil_image.load()
                pil_image.verify()

                if pil_image.format.lower() not in ["jpg", "jpeg", "png", "webp"]:
                    continue

                data = {
                    "id": sample_id,
                    "image": img_bytes,
                    "caption": text,
                    "caption_ls": [text],
                    "category": None,
                    "height": height,
                    "width": width,
                    "image_phash": image_phash,
                    "text_length": text_length,
                    "word_count": word_count,
                    "num_tokens_bert": num_tokens_bert,
                    "num_tokens_gpt": num_tokens_gpt,
                    "num_faces": num_faces,
                    "clip_similarity_vitb32": clip_similarity_vitb32,
                    "clip_similarity_vitl14": clip_similarity_vitl14,
                    "nsfw_score_opennsfw2": nsfw_score_opennsfw2,
                    "nsfw_score_gantman": nsfw_score_gantman,
                    "watermark_score": watermark_score,
                    "aesthetic_score_laion_v2": aesthetic_score_laion_v2,
                }
                finish_data_ls.append(data)
            return pa.Table.from_pylist(finish_data_ls)

        resizer = Resizer(
            image_size=256,
            resize_mode="no",
            min_image_size=10,
            resize_only_if_bigger=False,
        )

        idx_ = 0
        for idx, parquet_path in enumerate(filepath.values()):
            dataset = load_dataset("parquet", data_files=[parquet_path], split="train")

            parquet_path = Path(parquet_path)
            cache_file_path = parquet_path.parent.joinpath("coyo700m_cache_file", f"coyo700m-{idx}_cache_file.arrow")

            if not cache_file_path.parent.exists():
                print(f"mkdir coyo700m_cache_file at {cache_file_path.parent}")
                cache_file_path.parent.mkdir()

            dataset = dataset.map(
                image_downloader,
                num_proc=1,
                batched=True,
                batch_size=10,
                load_from_cache_file=True,
                desc=f"Coyo700m-{idx}",
                cache_file_name=str(cache_file_path),
                remove_columns=dataset.column_names,
            )
            for data in dataset:
                yield (idx_, data)
                idx_ += 1
