import os
from io import BytesIO
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Semaphore

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


class Laion400m(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [BuilderConfig(name="default", version="1.0.0", description=_DESCRIPTION)]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        features = Features(
            {
                "id": Value("int64"),
                "image": Image(),
                "caption": Value("string"),
                "caption_ls": [Value("string")],
                "category": Value("string"),
                "height": Value("int32"),
                "width": Value("int32"),
                "license": Value("string"),
                "nsfw": Value("string"),
                "similarity": Value("float"),
            }
        )

        self.resizer = Resizer(
            image_size=256,
            resize_mode="no",
            min_image_size=10,
            resize_only_if_bigger=False,
        )

        self.thread_num = os.getenv("LAION_THREAD_NUM", 15)
        self.num_proc = os.getenv("LAION_NUM_PROC", 20)
        self.batched = os.getenv("LAION_BATCHED", True)
        self.batch_size = os.getenv("LAION_BATCH_SIZE", 1000)

        return DatasetInfo(
            description=self.config.description,
            version=self.config.version,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
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
        idx_ = 0
        for idx, parquet_path in enumerate(filepath.values()):
            dataset = load_dataset("parquet", data_files=parquet_path, split="train")

            parquet_path = Path(parquet_path)
            cache_file_path = parquet_path.parent.joinpath("Laion400m_cache_file", f"Laion400m-{idx}_cache_file.arrow")

            if not cache_file_path.parent.exists():
                print(f"mkdir Laion400m_cache_file at {cache_file_path.parent}")
                cache_file_path.parent.mkdir()

            dataset = dataset.map(
                self.image_downloader,
                num_proc=self.num_proc,
                batched=self.batched,
                batch_size=self.batch_size,
                load_from_cache_file=True,
                desc=f"Laion400m-{idx}",
                cache_file_name=str(cache_file_path),
                remove_columns=dataset.column_names,
            )
            for data in dataset:
                yield (idx_, data)
                idx_ += 1

    def image_downloader(self, example):
        def downloader(data_row):
            sample_id, url, text, height, width, license_, nsfw, similarity = data_row
            idx, io_stream, err = download_image_with_retry(
                (sample_id, url),
                timeout=5,
                retries=2,
                user_agent_token=None,
                disallowed_header_directives=False,
            )
            if err:
                semaphore.release()
                return (None, None, None, None, None, None, None, None)

            img_bytes, _, _, orig_height, orig_width, err = self.resizer(io_stream)
            if height != orig_height or width != orig_width:
                semaphore.release()
                return (None, None, None, None, None, None, None, None)
            elif err:
                semaphore.release()
                return (None, None, None, None, None, None, None, None)

            semaphore.release()
            return sample_id, img_bytes, text, height, width, license_, nsfw, similarity

        def data_generator():
            for laion_row in laion_zip:
                semaphore.acquire()
                yield laion_row

        sample_id_ls, url_ls, text_ls, height_ls, width_ls, license_ls, nsfw_ls, similarity_ls = (
            example["SAMPLE_ID"] if isinstance(example["SAMPLE_ID"], list) else [example["SAMPLE_ID"]],
            example["URL"] if isinstance(example["URL"], list) else [example["URL"]],
            example["TEXT"] if isinstance(example["TEXT"], list) else [example["TEXT"]],
            example["HEIGHT"] if isinstance(example["HEIGHT"], list) else [example["HEIGHT"]],
            example["WIDTH"] if isinstance(example["WIDTH"], list) else [example["WIDTH"]],
            example["LICENSE"] if isinstance(example["LICENSE"], list) else [example["LICENSE"]],
            example["NSFW"] if isinstance(example["NSFW"], list) else [example["NSFW"]],
            example["similarity"] if isinstance(example["similarity"], list) else [example["similarity"]],
        )

        semaphore = Semaphore(self.thread_num * 2)
        loader = data_generator()
        finish_data_ls = list()
        laion_zip = zip(sample_id_ls, url_ls, text_ls, height_ls, width_ls, license_ls, nsfw_ls, similarity_ls)
        with ThreadPool(self.thread_num) as thread_pool:
            thead_iter = thread_pool.imap_unordered(downloader, loader)
            for sample_id, img_bytes, text, height, width, license_, nsfw, similarity in thead_iter:
                if not img_bytes:
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
                    "license": license_,
                    "nsfw": nsfw,
                    "similarity": similarity,
                }
                finish_data_ls.append(data)

        return pa.Table.from_pylist(finish_data_ls)
