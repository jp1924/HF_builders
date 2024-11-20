import json
import re
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Semaphore
from typing import List
from zipfile import ZipFile

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
    load_dataset,
)
from img2dataset.downloader import download_image_with_retry
from PIL import Image as PIL_Image
from tqdm import tqdm

from transformers import set_seed


set_seed(42)

_LICENSE = """Apache-2.0 license"""
_CITATION = """@misc{zheng2024multimodal,
      title={Multimodal Table Understanding}, 
      author={Mingyu Zheng and Xinwei Feng and Qingyi Si and Qiaoqiao She and Zheng Lin and Wenbin Jiang and Weiping Wang},
      year={2024},
      eprint={2406.08100},
      archivePrefix={arXiv},
      }
}"""
_HOMEPAGE = "https://github.com/SpursGoZmy/Table-LLaVA?tab=Apache-2.0-1-ov-file"


PRETRAINING_DATA = {
    "image_data": [
        "https://huggingface.co/datasets/SpursgoZmy/MMTab/resolve/main/MMTab-instruct_table_images_82K.zip",
        "https://huggingface.co/datasets/SpursgoZmy/MMTab/resolve/main/MMTab-pre_table_images_part_2_16K.zip",
        "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip",
    ],
    "label_data": "https://huggingface.co/datasets/SpursgoZmy/MMTab/resolve/main/enhanced_llava_pretrain_data_708K.json",
}

INSTRUCTION_DATA = {
    # "image_data": [
    #     "https://huggingface.co/datasets/SpursgoZmy/MMTab/resolve/main/MMTab-instruct_table_images_82K.zip",
    #     "http://images.cocodataset.org/zips/train2017.zip",
    #     "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
    #     "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
    #     "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
    #     "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
    # ],
    "label_data": "https://huggingface.co/datasets/SpursgoZmy/MMTab/resolve/main/enhanced_llava_sft_data_898K.json",
    "ocr_vqa": "https://drive.usercontent.google.com/download?id=1r0tyZUwGCc4wIG4RkiglCGNL_nFJjR6Q&export=download&authuser=0&confirm=t&uuid=cef68622-709d-4b1f-9b08-50b4de171e1b&at=AENtkXaP8DWSgw2R8DvZ-CWzSvV1%3A1731983081803",
}
EVALUATION_DATA = {
    "image_data": ["https://huggingface.co/datasets/SpursgoZmy/MMTab/resolve/main/MMTab-eval_table_images_23K.zip"],
    "label_data": "https://huggingface.co/datasets/SpursgoZmy/MMTab/resolve/main/MMTab-eval_test_data_49K.json",
}


_VERSION = "1.0.0"


class MMTab(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(name="pretraining", data_dir="pretraining", version=_VERSION),
        BuilderConfig(name="instruction", data_dir="instruction", version=_VERSION),
        BuilderConfig(name="evaluation", data_dir="evaluation", version=_VERSION),
    ]
    DEFAULT_CONFIG_NAME = "pretraining"
    DEFAULT_WRITER_BATCH_SIZE = 1000
    VERSION = _VERSION

    def _info(self):
        if self.config.name == "pretraining":
            features = Features(
                {
                    "id": Value("string"),
                    "image": Image(),
                    "conversations": [{"role": Value("string"), "content": Value("string")}],
                },
            )
        elif self.config.name == "instruction":
            features = Features(
                {
                    "id": Value("string"),
                    "image": Image(),
                    "conversations": [{"role": Value("string"), "content": Value("string")}],
                },
            )
        elif self.config.name == "evaluation":
            features = Features(
                {
                    "id": Value("string"),
                    "image": Image(),
                    "conversations": [
                        {
                            "role": Value("string"),
                            "content": Value("string"),
                        }
                    ],
                    "question": Value("string"),
                    "answer": Value("string"),
                    "answer_ls": Value("string"),
                    "dataset_name": Value("string"),
                    "original_query_type": Value("string"),
                }
            )

        else:
            raise ValueError()

        return DatasetInfo(
            features=features,
            license=_LICENSE,
            citation=_CITATION,
            homepage=_HOMEPAGE,
        )

    def _extract_zip_safely(self, zip_path: str, extract_path: Path) -> None:
        """ZIP 파일을 안전하게 압축 해제하는 메서드"""
        extract_path.mkdir(parents=True, exist_ok=True)

        with ZipFile(zip_path, "r") as zip_ref:
            for member in tqdm(zip_ref.namelist(), desc=f"Extracting {Path(zip_path).name}"):
                try:
                    zip_ref.extract(member, extract_path.as_posix())
                except Exception as e:
                    print(f"Warning: Failed to extract {member}: {e}")

    def _split_generators(self, dl_manager):
        def downloader(url_ls, save_path):
            save_path = Path(save_path)

            def data_generator():
                for url in url_ls:
                    semaphore.acquire()
                    yield url

            def down(url):
                _id = Path(url).stem
                idx, io_stream, err = download_image_with_retry(
                    (_id, url),
                    timeout=10,
                    retries=2,
                    user_agent_token=None,
                    disallowed_header_directives=False,
                )
                if err:
                    semaphore.release()
                    return None

                PIL_Image.open(io_stream).convert("RGB").save(save_path.joinpath(f"{_id}.jpg"))

                semaphore.release()
                return None

            loader = data_generator()
            semaphore = Semaphore(10 * 2)
            with ThreadPool(10) as thread_pool:
                thead_iter = thread_pool.imap_unordered(down, loader)
                for _ in thead_iter:
                    pass

            return {"url": [None] * len(url_ls)}

        task_ls = list()
        cache_dir = Path(dl_manager.download_config.cache_dir, "MMtabs")
        if self.config.name == "pretraining":
            # 다운로드
            data_path = dl_manager.download(PRETRAINING_DATA)
            save_dir = cache_dir.joinpath("pretraining")
            save_dir.mkdir(parents=True, exist_ok=True)

            label = json.loads(Path(data_path["label_data"]).read_text())

            # 압축 해제
            image_file_map = {}
            for zip_path in data_path["image_data"]:
                extract_dir = save_dir.joinpath(Path(zip_path).stem)
                if not extract_dir.exists():
                    self._extract_zip_safely(zip_path, extract_dir)

                image_file_map.update({path.stem: path.as_posix() for path in extract_dir.rglob("*.*")})

            for idx in tqdm(range(len(label))):
                label[idx]["image"] = image_file_map[Path(label[idx]["image"]).stem]

            task_ls.append(
                SplitGenerator(
                    name=Split.TRAIN,
                    gen_kwargs={
                        "label_ls": label,
                        "split": "train",
                    },
                )
            )
        elif self.config.name == "instruction":
            data_path = dl_manager.download(INSTRUCTION_DATA)
            save_dir = cache_dir.joinpath("instruction")
            save_dir.mkdir(parents=True, exist_ok=True)

            label = json.loads(Path(data_path["label_data"]).read_text())

            # 압축 해제
            image_file_map = {}
            for zip_path in data_path["image_data"]:
                extract_dir = save_dir.joinpath(Path(zip_path).stem)
                if not extract_dir.exists():
                    self._extract_zip_safely(zip_path, extract_dir)

                image_file_map.update({path.stem: path.as_posix() for path in extract_dir.rglob("*.*")})

            ocr_vqa = load_dataset("howard-hou/OCR-VQA", split="train")
            ocr_vqa_save_path = save_dir.joinpath("ocr_vqa")
            if not ocr_vqa_save_path.exists():
                ocr_vqa_save_path.mkdir(parents=True, exist_ok=True)
                for data_row in tqdm(ocr_vqa):
                    file_path = ocr_vqa_save_path.joinpath(f'{data_row["image_id"]}.jpg')
                    data_row["image"].convert("RGB").save(file_path.as_posix())

            image_file_map.update({path.stem: path.as_posix() for path in ocr_vqa_save_path.rglob("*.*")})

            # ocr_vqa_save_path = save_dir.joinpath("ocr_vqa")
            # if not ocr_vqa_save_path.exists():
            #     ocr_vqa_save_path.mkdir(parents=True, exist_ok=True)
            #     ocr_vqa_ls = json.loads(Path(data_path["ocr_vqa"]).read_text())
            #     dataset = Dataset.from_dict(
            #         {
            #             "url": [row["imageURL"] for row in ocr_vqa_ls.values()],
            #             "id": [row["id"] for row in ocr_vqa_ls.values()],
            #         },
            #     )
            #     dataset.map(
            #         downloader,
            #         num_proc=10,
            #         input_columns=["id", "url"],
            #         batch_size=1000,
            #         batched=True,
            #         fn_kwargs={"save_path": ocr_vqa_save_path.as_posix()},
            #     )

            for idx in tqdm(range(len(label))):
                if "image" not in label[idx]:
                    continue
                stem = Path(label[idx]["image"]).stem
                if stem not in image_file_map:
                    continue
                label[idx]["image"] = image_file_map[stem]

            task_ls.append(
                SplitGenerator(
                    name=Split.TRAIN,
                    gen_kwargs={
                        "label_ls": label,
                        "split": "train",
                    },
                )
            )
        elif self.config.name == "evaluation":
            data_path = dl_manager.download(EVALUATION_DATA)
            save_dir = cache_dir.joinpath("evaluation")
            save_dir.mkdir(parents=True, exist_ok=True)

            label = json.loads(Path(data_path["label_data"]).read_text())

            image_file_map = {}
            for zip_path in data_path["image_data"]:
                extract_dir = save_dir.joinpath(Path(zip_path).stem)
                if not extract_dir.exists():
                    self._extract_zip_safely(zip_path, extract_dir)

                image_file_map.update({path.stem: path.as_posix() for path in extract_dir.rglob("*.*")})

            for idx in tqdm(range(len(label))):
                label[idx] = {
                    "id": label[idx]["item_id"],
                    "image": image_file_map[label[idx]["image_id"]],
                    "conversations": [
                        {"from": "Human", "value": f'<image>{label[idx]["input"]}'},
                        {"from": "gpt", "value": label[idx]["output"]},
                    ],
                    "question": label[idx]["input"],
                    "answer": label[idx]["output"],
                    "answer_ls": json.dumps(label[idx]["answer_list"]),
                    "dataset_name": label[idx]["task_type"],
                    "original_query_type": label[idx]["original_query_type"],
                }

            task_ls.append(
                SplitGenerator(
                    name=Split.VALIDATION,
                    gen_kwargs={
                        "label_ls": label,
                        "split": "valid",
                    },
                )
            )

        return task_ls

    def _generate_examples(self, label_ls: List, split: str):
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

        for idx, label in enumerate(label_ls):
            new_conversations_ls = list()
            if "image" not in label:
                del label["model"]

                label["image"] = None
                for turn in label["conversations"]:
                    role = "assistant" if turn["from"] == "gpt" else "user"
                    content = turn["value"]
                    new_conversations_ls.append({"role": role, "content": content})
            else:
                for turn in label["conversations"]:
                    role = "assistant" if turn["from"] == "gpt" else "user"
                    content = json.dumps(convert_mm_content(turn["value"], "<image>"), ensure_ascii=False)
                    new_conversations_ls.append({"role": role, "content": content})

                img_path = Path(label["image"])
                if not img_path.exists():
                    print(label["image"], "가 존재하지 않음. ")
                    continue
                label["image"] = PIL_Image.open(img_path.as_posix())

            label["conversations"] = new_conversations_ls

            yield (idx, label)
