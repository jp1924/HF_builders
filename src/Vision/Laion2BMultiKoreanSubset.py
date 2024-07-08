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


_URLs = {
    "157cf2686bec271f": "https://huggingface.co/datasets/Bingsu/laion2B-multi-korean-subset/resolve/main/data/train-00000-of-00006-5fe89370890da24f.parquet?download=true",
    "27c90b30911ef7aa": "https://huggingface.co/datasets/Bingsu/laion2B-multi-korean-subset/resolve/main/data/train-00001-of-00006-aab9b1bf52a2df3f.parquet?download=true",
    "827d4088a57b200a": "https://huggingface.co/datasets/Bingsu/laion2B-multi-korean-subset/resolve/main/data/train-00002-of-00006-1c4b4049541a99d4.parquet?download=true",
    "52d7cd2b0f7eadf4": "https://huggingface.co/datasets/Bingsu/laion2B-multi-korean-subset/resolve/main/data/train-00003-of-00006-9f022966b6bb9cf8.parquet?download=true",
    "12f9b17a7d230ec9": "https://huggingface.co/datasets/Bingsu/laion2B-multi-korean-subset/resolve/main/data/train-00004-of-00006-829cbdc252d37dd4.parquet?download=true",
    "561e693a58593192": "https://huggingface.co/datasets/Bingsu/laion2B-multi-korean-subset/resolve/main/data/train-00005-of-00006-5435e409ba6a048d.parquet?download=true",
}
_DESCRIPTION = """a subset data of laion/laion2B-multi, including only korean"""
_VERSION = Version("1.0.0")


class Laion2BMultiKoreanSubset(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [BuilderConfig(name="default", version=_VERSION, description=_DESCRIPTION)]
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
                "language": Value("string"),
                "height": Value("int32"),
                "width": Value("int32"),
                "nsfw": Value("string"),
                "license": Value("string"),
                "similarity": Value("float32"),
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
            description=_DESCRIPTION,
            version=_VERSION,
            features=features,
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
        idx_ = 0
        for idx, parquet_path in enumerate(filepath.values()):
            dataset = load_dataset("parquet", data_files=parquet_path, split="train")

            parquet_path = Path(parquet_path)
            cache_file_path = parquet_path.parent.joinpath(
                "Laion2BMultiKoreanSubset_cache_file", f"Laion2BMultiKoreanSubset-{idx}_cache_file.arrow"
            )

            if not cache_file_path.parent.exists():
                print(f"mkdir Laion2BMultiKoreanSubset_cache_file at {cache_file_path.parent}")
                cache_file_path.parent.mkdir()

            dataset = dataset.map(
                self.image_downloader,
                num_proc=self.num_proc,
                batched=self.batched,
                batch_size=self.batch_size,
                load_from_cache_file=True,
                desc=f"Laion2BMultiKoreanSubset-{idx}",
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
