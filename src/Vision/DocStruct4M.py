import json
import os
import re
import warnings
from io import BytesIO
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Semaphore

from datasets import (
    BuilderConfig,
    Dataset,
    DatasetInfo,
    Features,
    GeneratorBasedBuilder,
    Image,
    Split,
    SplitGenerator,
    Value,
    Version,
)
from datasets.config import DEFAULT_MAX_BATCH_SIZE
from img2dataset.resizer import Resizer
from natsort import natsorted
from PIL import Image as PIL_Image
from tqdm import tqdm

from transformers.trainer_pt_utils import get_length_grouped_indices


TRAIN_IMG_URLS = {
    "partial-imgs.00": "https://huggingface.co/datasets/mPLUG/DocStruct4M/resolve/main/partial-imgs.00",
    "partial-imgs.01": "https://huggingface.co/datasets/mPLUG/DocStruct4M/resolve/main/partial-imgs.01",
    "partial-imgs.02": "https://huggingface.co/datasets/mPLUG/DocStruct4M/resolve/main/partial-imgs.02",
    "partial-imgs.03": "https://huggingface.co/datasets/mPLUG/DocStruct4M/resolve/main/partial-imgs.03",
    "partial-imgs.04": "https://huggingface.co/datasets/mPLUG/DocStruct4M/resolve/main/partial-imgs.04",
    "partial-imgs.05": "https://huggingface.co/datasets/mPLUG/DocStruct4M/resolve/main/partial-imgs.05",
    "partial-imgs.06": "https://huggingface.co/datasets/mPLUG/DocStruct4M/resolve/main/partial-imgs.06",
    "partial-imgs.07": "https://huggingface.co/datasets/mPLUG/DocStruct4M/resolve/main/partial-imgs.07",
}
TRAIN_LABEL_URL = "https://huggingface.co/datasets/mPLUG/DocStruct4M/resolve/main/struct_aware_parse.jsonl"

VALID_IMG_URL = "https://huggingface.co/datasets/mPLUG/DocStruct4M/resolve/main/val_imgs.tar.gz"
VALID_LABEL_URL = "https://huggingface.co/datasets/mPLUG/DocStruct4M/resolve/main/val.jsonl"


_HOMEPAGE = "https://huggingface.co/datasets/mPLUG/M-Paper"
_LICENSE = "Apache License 2.0"
_CITATION = """@article{hu2023paperowl,
  title={mplug-paperowl: Scientific diagram analysis with the multimodal large language model},
  author={Hu, Anwen and Shi, Yaya and Xu, Haiyang and Ye, Jiabo and Ye, Qinghao and Yan, Ming and Li, Chenliang and Qian, Qi and Zhang, Ji and Huang, Fei},
  journal={arXiv preprint arXiv:2311.18248},
  year={2023}
}"""
_DESCRIPTION = """M-Paper is a Scientific Diagram Analysis dataset based on 48k high-quality arxiv papers (2021~2023) on Machine Learning. M-Paper contains 447k diagram images and supports 3 tasks: Diagram Captioning, Diagram Analysis and Outline Recommendation."""
_VERSION = Version("1.0.0")


class DocStruct4M(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [BuilderConfig(name="default", version=_VERSION)]
    DEFAULT_CONFIG_NAME = "default"
    VERSION = _VERSION

    def _info(self):
        self.features = Features(
            {
                "id": Value("int64"),
                "image": Image(),
                "conversations": [{"role": Value("string"), "content": Value("string")}],
                "dataset_name": Value("string"),
                "task_name": Value("string"),
            }
        )

        self.resizer = Resizer(
            image_size=256,
            resize_mode="no",
            min_image_size=10,
            resize_only_if_bigger=False,
        )

        self.thread_num = int(os.getenv("DocStruct4M_THREAD_NUM", "20"))
        self.num_proc = int(os.getenv("DocStruct4M_NUM_PROC", "20"))
        self.batched = int(os.getenv("DocStruct4M_BATCHED", True))
        self.batch_size = int(os.getenv("DocStruct4M_BATCH_SIZE", "100"))

        return DatasetInfo(
            description=_DESCRIPTION,
            features=self.features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            version=_VERSION,
        )

    def _split_generators(self, dl_manager):
        cache_dir = Path(dl_manager.download_config.cache_dir)
        train_partial_img_paths = dl_manager.download(TRAIN_IMG_URLS)
        train_label_path = dl_manager.download(TRAIN_LABEL_URL)

        valid_img_paths = dl_manager.download_and_extract(VALID_IMG_URL)
        valid_label_path = dl_manager.download(VALID_LABEL_URL)

        shard_zip_dataset = natsorted(list(train_partial_img_paths.items()), key=lambda x: x[0])
        download_path = cache_dir.joinpath("DocStruct4M.tar.gz")
        try:
            extract_path = dl_manager.extract(download_path)
        except BaseException as e:  # noqa: F841
            partial_paths = " ".join([x[1] for x in shard_zip_dataset])
            os.system(f"cat {partial_paths} > {download_path.as_posix()}")

            # 이 방법으로 하니깐 OOM 뜨더라.
            # with open(download_path, "wb") as byte_f:
            #     for _, part_path_ls in tqdm(shard_zip_dataset, desc="concat"):
            #         part_bytes = Path(part_path_ls).read_bytes()
            #         byte_f.write(part_bytes)
            extract_path = dl_manager.extract(download_path)

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "image_path": Path(extract_path),
                    "label_path": Path(train_label_path),
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "label_path": Path(valid_label_path),
                    "image_path": Path(valid_img_paths),
                },
            ),
        ]

    def _generate_examples(self, label_path: Path, image_path: Path):
        def load_image_file(example, img_dir_path):
            def convert_mm_content(content: str, img_token: str):
                img_split_regex = re.compile(rf"{img_token}|.")

                new_content_ls = list()
                sentence = ""
                for token in img_split_regex.findall(content):
                    if re.match(img_token, token):
                        if sentence:
                            new_content_ls.append({"type": "text", "text": sentence})
                            sentence = ""
                        new_content_ls.append({"type": "image"})
                        continue

                    sentence += token
                else:
                    if sentence:
                        new_content_ls.append({"type": "text", "text": sentence})

                return new_content_ls

            def downloader(data_row):
                id_, image_ls, conversations, dataset_name, task_name = data_row

                if len(image_ls) > 1:
                    return None, None, None, None, None, None

                loaded_image_ls = list()
                loaded_image_size_ls = list()
                for img_path in image_ls:
                    try:
                        with warnings.catch_warnings(record=True) as warn_ls:
                            img_bytes = Path(img_dir_path, img_path).read_bytes()
                            pil_image = PIL_Image.open(BytesIO(img_bytes))
                            pil_image.verify()

                            if warn_ls:
                                # DecompressionBombWarning이나 파일 일부가 손상된 경우 warn이 발생함.
                                # 특히 DecompressionBombWarning는 이미지가 너무 커서 문제가 발생하는 경우임.
                                # 이미지 저장 및 업로드 시 문제가 발생하기 때문에 건너뜀.
                                for warn in warn_ls:
                                    print(f"{warn.message}가 발생함. 해당 샘플은 skip함.")
                                semaphore.release()
                                return None, None, None, None, None, None
                        
                        if pil_image.format.lower() not in ["jpg", "jpeg", "png", "webp"]:
                            semaphore.release()
                            return None, None, None, None, None, None

                        loaded_image_ls.append(img_bytes)
                        loaded_image_size_ls.append(pil_image.width * pil_image.height)
                    except BaseException as e:  # noqa: F841
                        print(f"{e}가 발생함. 해당 샘플은 skip함.")
                        semaphore.release()
                        return None, None, None, None, None, None

                img_token_num = 0
                new_conversation_ls = list()
                for chat in conversations:
                    content = convert_mm_content(chat["content"], r"<\|image\|>")

                    img_token_num += len([chat for chat in content if chat["type"] == "image"])
                    
                    chat = {"role": chat["role"], "content": json.dumps(content, ensure_ascii=False)}
                    new_conversation_ls.append(chat)

                if img_token_num != len(loaded_image_ls):
                    semaphore.release()
                    return None, None, None, None, None, None

                semaphore.release()
                return id_, loaded_image_ls[0], new_conversation_ls, dataset_name, task_name, sum(loaded_image_size_ls)

            def data_generator():
                for laion_row in DocStruct4M_zip:
                    semaphore.acquire()
                    yield laion_row

            id_ls, image_ls, messages_ls, dataset_name_ls, task_name_ls = (
                example["id"] if isinstance(example["id"], list) else [example["id"]],
                example["image"] if isinstance(example["image"], list) else [example["image"]],
                example["messages"] if isinstance(example["messages"], list) else [example["messages"]],
                example["dataset_name"] if isinstance(example["dataset_name"], list) else [example["dataset_name"]],
                example["task_name"] if isinstance(example["task_name"], list) else [example["task_name"]],
            )

            finish_id_ls = list()
            finish_image_ls = list()
            finish_image_size_ls = list()
            finish_conversations_ls = list()
            finish_dataset_name_ls = list()
            finish_task_name_ls = list()

            semaphore = Semaphore(self.thread_num * 2)
            loader = data_generator()
            DocStruct4M_zip = zip(id_ls, image_ls, messages_ls, dataset_name_ls, task_name_ls)
            with ThreadPool(self.thread_num) as thread_pool:
                thead_iter = thread_pool.imap_unordered(downloader, loader)
                for id_, img_ls, conversations, dataset_name, task_name, img_size in thead_iter:
                    if not img_ls:
                        continue

                    finish_id_ls.append(id_)
                    finish_image_ls.append(img_ls)
                    finish_conversations_ls.append(conversations)
                    finish_dataset_name_ls.append(dataset_name)
                    finish_task_name_ls.append(task_name)
                    finish_image_size_ls.append(img_size)

            return {
                "id": finish_id_ls,
                "image": finish_image_ls,
                "conversations": finish_conversations_ls,
                "dataset_name": finish_dataset_name_ls,
                "task_name": finish_task_name_ls,
                "image_size": finish_image_size_ls,
            }

        # 컬럼이 일정하지 않아서 load_dataset으로 불러오지 못함. 데이터에서 rvl_cdip_class를 검색해 보셈.
        labels = list()
        with open(str(label_path), "r", encoding="utf-8") as f:
            lines = f.readlines()
            for idx, line in enumerate(tqdm(lines, desc="load_labels")):
                line = json.loads(line.strip())
                line["id"] = idx
                labels.append(line)

        datasets = Dataset.from_list(labels)
        datasets = datasets.map(
            load_image_file,
            num_proc=self.num_proc,
            batched=self.batched,
            batch_size=self.batch_size,
            load_from_cache_file=True,
            remove_columns=datasets.column_names,
            fn_kwargs={"img_dir_path": image_path},
        )
        image_size_ls = get_length_grouped_indices(
            datasets["image_size"],
            batch_size=1,
            mega_batch_mult=DEFAULT_MAX_BATCH_SIZE,
        )

        idx_ = 0
        for idx in image_size_ls:
            data = datasets[idx]
            del data["image_size"]
            yield (idx_, data)
            idx_ += 1
