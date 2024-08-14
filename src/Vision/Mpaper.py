import json
import os
import warnings
from io import BytesIO
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Semaphore

import pyarrow as pa
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
    "partial-imgs.00": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.00?download=true",
    "partial-imgs.01": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.01?download=true",
    "partial-imgs.02": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.02?download=true",
    "partial-imgs.03": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.03?download=true",
    "partial-imgs.04": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.04?download=true",
    "partial-imgs.05": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.05?download=true",
    "partial-imgs.06": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.06?download=true",
    "partial-imgs.07": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.07?download=true",
    "partial-imgs.08": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.08?download=true",
    "partial-imgs.09": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.09?download=true",
    "partial-imgs.10": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.10?download=true",
    "partial-imgs.11": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.11?download=true",
    "partial-imgs.12": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.12?download=true",
    "partial-imgs.13": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.13?download=true",
    "partial-imgs.14": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.14?download=true",
    "partial-imgs.15": "https://huggingface.co/datasets/mPLUG/M-Paper/resolve/main/partial-imgs.15?download=true",
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
        self.batched = int(os.getenv("Mpaper_BATCHED", True))
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
        except:
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
        datasets = Dataset.from_json(str(label_path))
        image_size_ls = [
            sum([image_path.joinpath(pth).stat().st_size for pth in pth_ls]) if pth_ls else 0
            for pth_ls in tqdm(datasets["image"], desc="estimate_size")
        ]
        # build시 한 arrow 파일에 과도한 용량이 집중되면, arrow로 저장이 되질 않을때가 있음.
        # 애러를 방지하기 위해 arrow 파일마다 용량을 고르게 분배해서 부하를 피하기 위한 방법
        image_size_ls = get_length_grouped_indices(
            image_size_ls,
            batch_size=1,
            mega_batch_mult=DEFAULT_MAX_BATCH_SIZE,
        )
        datasets = datasets.map(
            self.load_image_file,
            num_proc=self.num_proc,
            batched=self.batched,
            batch_size=self.batch_size,
            features=self.features,
            load_from_cache_file=True,
            remove_columns=datasets.column_names,
            fn_kwargs={"img_dir_path": image_path},
        )
        idx_ = 0
        for idx in image_size_ls:
            yield (idx_, datasets[idx])
            idx_ += 1

    def load_image_file(self, example, img_dir_path):
        def downloader(data_row):
            sample_id, image_ls, conversations = data_row

            loaded_image_ls = list()
            for img_path in image_ls:
                try:
                    img_bytes = Path(img_dir_path, img_path).read_bytes()
                    pil_image = PIL_Image.open(BytesIO(img_bytes))
                    pil_image.verify()

                    if pil_image.format.lower() not in ["jpg", "jpeg", "png", "webp"]:
                        semaphore.release()
                        return None, None, None

                    loaded_image_ls.append(img_bytes)
                except:
                    semaphore.release()
                    return None, None, None

            conversation_ls = list()
            for chat in conversations:
                if "<image>" in chat["value"] and chat["from"] == "user":
                    img_split_chat = chat["value"].split("<image>")
                    img_split_chat = [txt.replace("<|context|>: ", "").strip() for txt in img_split_chat]
                    img_split_chat = [txt for txt in img_split_chat if txt]

                    new_chat_content_ls = list()
                    new_chat_content_ls.append({"type": "text", "text": img_split_chat[0]})
                    for idx in range(1, len(img_split_chat)):
                        txt = img_split_chat[idx]

                        new_chat_content_ls.append({"type": "image"})
                        new_chat_content_ls.append({"type": "text", "text": txt})

                    chat["value"] = json.dumps(new_chat_content_ls, ensure_ascii=False)
                elif chat["from"] == "assistant" and "<image>" in chat["value"]:
                    warnings.warn(
                        "assistant에 img가 있음! 확인 바람.",
                        f"""{chat["value"]}""",
                    )

                chat = {"role": chat["from"], "content": chat["value"]}
                conversation_ls.append(chat)

            semaphore.release()
            return sample_id, loaded_image_ls, conversation_ls

        def data_generator():
            for laion_row in Mpaper_zip:
                semaphore.acquire()
                yield laion_row

        sample_id_ls, image_ls, conversations_ls = (
            example["id"] if isinstance(example["id"], list) else [example["id"]],
            example["image"] if isinstance(example["image"], list) else [example["image"]],
            example["conversations"] if isinstance(example["conversations"], list) else [example["conversations"]],
        )

        semaphore = Semaphore(self.thread_num * 2)
        loader = data_generator()
        finish_data_ls = list()
        Mpaper_zip = zip(sample_id_ls, image_ls, conversations_ls)
        with ThreadPool(self.thread_num) as thread_pool:
            thead_iter = thread_pool.imap_unordered(downloader, loader)
            for sample_id, img_ls, conversations in thead_iter:
                data = {
                    "id": sample_id,
                    "image": img_ls,
                    "conversations": conversations,
                }
                finish_data_ls.append(data)

        return pa.Table.from_pylist(finish_data_ls)
