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
    load_dataset,
)
from img2dataset.downloader import download_image_with_retry
from img2dataset.resizer import Resizer
from PIL import Image as PIL_Image


_HOMEPAGE = "https://huggingface.co/datasets/QuoQA-NLP/KoCC3M"


_DATANAME = "KoCC3M"


_DESCRIPTION = """CC3M of flax-community/conceptual-captions-12 translated from English to Korean."""


TRAIN_URLs = {
    "157cf2686bec271f": "https://huggingface.co/datasets/QuoQA-NLP/KoCC3M/resolve/main/data/train-00000-of-00002-cc8e11261b9f26e1.parquet?download=true",
    "27c90b30911ef7aa": "https://huggingface.co/datasets/QuoQA-NLP/KoCC3M/resolve/main/data/train-00001-of-00002-3d1333b77c91c8c1.parquet?download=true",
}
VALID_URLs = {
    "27c90b30911ef7aa": "https://huggingface.co/datasets/QuoQA-NLP/KoCC3M/resolve/main/data/validation-00000-of-00001-168f14d7fd7256ba.parquet?download=true",
}


class KoCC3M(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [BuilderConfig(name="default", version="1.0.0", description=_DESCRIPTION)]
    DEFAULT_CONFIG_NAME = "default"

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

        self.resizer = Resizer(
            image_size=256,
            resize_mode="no",
            min_image_size=10,
            resize_only_if_bigger=False,
        )

        self.thread_num = os.getenv("CC3_THREAD_NUM", 20)
        self.num_proc = os.getenv("CC3_NUM_PROC", 20)
        self.batched = os.getenv("CC3_BATCHED", True)
        self.batch_size = os.getenv("CC3_BATCH_SIZE", 100)

        return DatasetInfo(
            description=self.config.description,
            version=self.config.version,
            features=features,
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        train_file_path = dl_manager.download_and_extract(TRAIN_URLs)
        valid_file_path = dl_manager.download_and_extract(VALID_URLs)
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "filepath": train_file_path,
                    "split": "train",
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "filepath": valid_file_path,
                    "split": "valid",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        idx_ = 0
        for idx, parquet_path in enumerate(filepath.values()):
            dataset = load_dataset("parquet", data_files=parquet_path, split="train")

            parquet_path = Path(parquet_path)
            cache_file_path = parquet_path.parent.joinpath(
                "KoCC3M_cache_file", f"KoCC3M-{split}-{idx}_cache_file.arrow"
            )

            if not cache_file_path.parent.exists():
                print(f"mkdir KoCC3M_cache_file at {cache_file_path.parent}")
                cache_file_path.parent.mkdir()

            dataset = dataset.map(
                self.image_downloader,
                num_proc=self.num_proc,
                batched=self.batched,
                batch_size=self.batch_size,
                load_from_cache_file=True,
                desc=f"KoCC3M-{idx}",
                cache_file_name=str(cache_file_path),
                remove_columns=dataset.column_names,
            )
            for data in dataset:
                data["id"] = idx_
                yield (idx_, data)
                idx_ += 1

    def image_downloader(self, example):
        def downloader(data_row):
            url, korean_caption, english_caption = data_row
            _, io_stream, err = download_image_with_retry(
                (0, url),
                timeout=5,
                retries=2,
                user_agent_token=None,
                disallowed_header_directives=False,
            )

            img_bytes, _, _, orig_height, orig_width, err = self.resizer(io_stream)
            if not (orig_height and orig_width):
                semaphore.release()
                return (None, None, None)
            elif orig_height < 10 or orig_width < 10:
                semaphore.release()
                return (None, None, None)
            elif err:
                semaphore.release()
                return (None, None, None)

            semaphore.release()
            return img_bytes, korean_caption, english_caption

        def data_generator():
            for laion_row in cc12_zip:
                semaphore.acquire()
                yield laion_row

        url_ls, korean_caption_ls, english_caption_ls = (
            example["image_url"] if isinstance(example["image_url"], list) else [example["image_url"]],
            example["korean_caption"] if isinstance(example["korean_caption"], list) else [example["korean_caption"]],
            example["english_caption"]
            if isinstance(example["english_caption"], list)
            else [example["english_caption"]],
        )

        semaphore = Semaphore(self.thread_num * 2)
        loader = data_generator()
        finish_data_ls = list()
        cc12_zip = zip(url_ls, korean_caption_ls, english_caption_ls)
        with ThreadPool(self.thread_num) as thread_pool:
            thead_iter = thread_pool.imap_unordered(downloader, loader)
            for img_bytes, korean_caption, english_caption in thead_iter:
                if not img_bytes:
                    continue

                pil_image = PIL_Image.open(BytesIO(img_bytes))
                pil_image.load()
                pil_image.verify()

                if pil_image.format.lower() not in ["jpg", "jpeg", "png", "webp"]:
                    continue

                data = {
                    "image": img_bytes,
                    "caption": korean_caption,
                    "caption_ls": [korean_caption],
                    "category": None,
                    "en_caption": english_caption,
                }
                finish_data_ls.append(data)

        return pa.Table.from_pylist(finish_data_ls)
