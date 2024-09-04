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


IMG_URLS = {
    "partial-imgs.00": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.00",
    "partial-imgs.01": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.01",
    "partial-imgs.02": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.02",
    "partial-imgs.03": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.03",
    "partial-imgs.04": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.04",
    "partial-imgs.05": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.05",
    "partial-imgs.06": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.06",
    "partial-imgs.07": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.07",
    "partial-imgs.08": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.08",
    "partial-imgs.09": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.09",
    "partial-imgs.10": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.10",
    "partial-imgs.11": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.11",
    "partial-imgs.12": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.12",
    "partial-imgs.13": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.13",
    "partial-imgs.14": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.14",
    "partial-imgs.15": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.15",
}

LABEL_URLS = {
    "3tasks_train.jsonl": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/sft/3tasks_train.jsonl",
    "3tasks_test.jsonl": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/sft/3tasks_test.jsonl",
    "3tasks_val.jsonl": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/sft/3tasks_val.jsonl",
}

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


class MPaper(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [BuilderConfig(name="default", version=_VERSION)]
    DEFAULT_CONFIG_NAME = "default"
    VERSION = _VERSION

    def _info(self):
        self.features = Features(
            {
                "id": Value("int64"),
                "image": [Image()],
                "conversations": [{"role": Value("string"), "content": Value("string")}],
            }
        )

        self.resizer = Resizer(
            image_size=256,
            resize_mode="no",
            min_image_size=10,
            resize_only_if_bigger=False,
        )

        self.thread_num = int(os.getenv("Mpaper_THREAD_NUM", "20"))
        self.num_proc = int(os.getenv("Mpaper_NUM_PROC", "20"))
        self.batched = bool(os.getenv("Mpaper_BATCHED", True))
        self.batch_size = int(os.getenv("Mpaper_BATCH_SIZE", "100"))

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
        partial_img_paths = dl_manager.download(IMG_URLS)
        label_path = dl_manager.download(LABEL_URLS)

        shard_zip_dataset = natsorted(list(partial_img_paths.items()), key=lambda x: x[0])
        download_path = cache_dir.joinpath("Mpaper.tar.gz")
        try:
            extract_path = dl_manager.extract(download_path)
        except BaseException as e:  # noqa: F841
            with open(download_path, "wb") as byte_f:
                for _, part_path_ls in tqdm(shard_zip_dataset, desc="concat"):
                    part_bytes = Path(part_path_ls).read_bytes()
                    byte_f.write(part_bytes)
            extract_path = dl_manager.extract(download_path)

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "label_path": Path(label_path["3tasks_train.jsonl"]),
                    "image_path": Path(extract_path),
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "label_path": Path(label_path["3tasks_val.jsonl"]),
                    "image_path": Path(extract_path),
                },
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "label_path": Path(label_path["3tasks_test.jsonl"]),
                    "image_path": Path(extract_path),
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
                sample_id, image_ls, conversations = data_row

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
                                return None, None, None, None
                        if pil_image.format.lower() not in ["jpg", "jpeg", "png", "webp"]:
                            semaphore.release()
                            return None, None, None, None

                        loaded_image_ls.append(img_bytes)
                        loaded_image_size_ls.append(pil_image.width * pil_image.height)
                    except BaseException as e:  # noqa: F841
                        print(f"{e}가 발생함. 해당 샘플은 skip함.")
                        semaphore.release()
                        return None, None, None, None

                img_token_num = 0
                new_conversation_ls = list()
                for chat in conversations:
                    value = chat["value"].replace("<|context|>: ", "")
                    content = convert_mm_content(value, r"<image>")

                    img_token_num += len([chat for chat in content if chat["type"] == "image"])

                    chat = {"role": chat["from"], "content": json.dumps(content, ensure_ascii=False)}
                    new_conversation_ls.append(chat)

                if img_token_num != len(loaded_image_ls):
                    semaphore.release()
                    return None, None, None, None

                semaphore.release()
                return sample_id, loaded_image_ls, new_conversation_ls, sum(loaded_image_size_ls)

            def data_generator():
                for laion_row in Mpaper_zip:
                    semaphore.acquire()
                    yield laion_row

            sample_id_ls, image_ls, conversations_ls = (
                example["id"] if isinstance(example["id"], list) else [example["id"]],
                example["image"] if isinstance(example["image"], list) else [example["image"]],
                example["conversations"] if isinstance(example["conversations"], list) else [example["conversations"]],
            )

            finish_id_ls = list()
            finish_image_ls = list()
            finish_image_size_ls = list()
            finish_conversations_ls = list()

            semaphore = Semaphore(self.thread_num * 2)
            loader = data_generator()
            Mpaper_zip = zip(sample_id_ls, image_ls, conversations_ls)
            with ThreadPool(self.thread_num) as thread_pool:
                thead_iter = thread_pool.imap_unordered(downloader, loader)
                for sample_id, img_ls, conversations, img_size in thead_iter:
                    if not img_ls:
                        continue

                    finish_id_ls.append(sample_id)
                    finish_image_ls.append(img_ls)
                    finish_conversations_ls.append(conversations)
                    finish_image_size_ls.append(img_size)

            return {
                "id": finish_id_ls,
                "image": finish_image_ls,
                "image_size": finish_image_size_ls,
                "conversations": finish_conversations_ls,
            }

        datasets = Dataset.from_json(str(label_path))
        datasets = datasets.map(
            load_image_file,
            num_proc=self.num_proc,
            batched=self.batched,
            batch_size=self.batch_size,
            load_from_cache_file=True,
            remove_columns=datasets.column_names,
            fn_kwargs={"img_dir_path": image_path},
        )

        # build시 한 arrow 파일에 과도한 용량이 집중되면, arrow로 저장이 되질 않을때가 있음.
        # 애러를 방지하기 위해 arrow 파일마다 용량을 고르게 분배해서 부하를 피하기 위한 방법
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
