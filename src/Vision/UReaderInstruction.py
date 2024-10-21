import json
import os
import re
import warnings
from io import BytesIO
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Semaphore
from typing import List

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
    concatenate_datasets,
)
from datasets.config import DEFAULT_MAX_BATCH_SIZE
from PIL import Image as PIL_Image

from transformers.trainer_pt_utils import get_length_grouped_indices


URLS = {
    "WikiTableQuestions": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/WikiTableQuestions.tar",
    "InfographicsVQA": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/InfographicsVQA.tar",
    "KleisterCharity": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/KleisterCharity.tar",
    "benchmark_files": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/benchmark_files.zip",
    "ureader_json": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/ureader_json.zip",
    "VisualMRC": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/VisualMRC.tar",
    "DeepForm": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/DeepForm.tar",
    "TextCaps": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/TextCaps.tar",
    "TabFact": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/TabFact.tar",
    "ChartQA": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/ChartQA.tar",
    "TextVQA": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/TextVQA.tar",
    "DocVQA": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/DocVQA.tar",
}
_VERSION = Version("1.1.0")


class UReaderInstruction(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [BuilderConfig(name="default", version=_VERSION)]
    DEFAULT_CONFIG_NAME = "default"
    VERSION = _VERSION

    def _info(self):
        features = Features(
            {
                "id": Value("string"),
                "image": Image(),
                "conversations": [{"role": Value("string"), "content": Value("string")}],
                "metadata": {"source": Value("string")},
            }
        )
        self.thread_num = int(os.getenv("UReaderInstruct_THREAD_NUM", "10"))
        self.num_proc = int(os.getenv("UReaderInstruct_NUM_PROC", "10"))
        self.batched = bool(os.getenv("UReaderInstruct_BATCHED", True))
        self.batch_size = int(os.getenv("UReaderInstruct_BATCH_SIZE", "100"))
        self.shard_num = int(os.getenv("UReaderInstruct_SHARD_NUM", "4"))

        return DatasetInfo(
            features=features,
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        def matching_path(prefix: str):
            data_ls = list()
            for jsonl in label_jsonl_ls:
                if prefix not in jsonl.stem:
                    continue

                # test_TabFact > .split("_")[-1] > TabFact
                data_name = jsonl.stem.split("_")[-1]

                jsonl = jsonl.read_text().split("\n")
                json_ls = [json.loads(line) for line in jsonl if line]
                for idx, line in enumerate(json_ls):
                    img_ls = list()
                    for img in line["image"]:
                        start_name = img.replace("DUE_Benchmark/", "").split("/")[0]
                        data_dir = (
                            downloaded_files[data_name]
                            if data_name in downloaded_files
                            else downloaded_files[start_name]
                        )
                        img_ls.append(Path(data_dir, img).as_posix())
                    line["image"] = img_ls
                    line["date_from"] = start_name

                    # NOTE: conversations의 img 토큰이 서로 다른 chat으로 분리되어 있어서 이걸 하나로 합침.
                    #       모든 conversations의 앞에 img token이 있음을 확인함.
                    conversations = line["conversations"][1:]
                    conversations[0]["value"] = f"""<image>{conversations[0]['value']}"""
                    line["conversations"] = conversations

                    json_ls[idx] = line

                data_ls.extend(json_ls)
            return data_ls

        downloaded_files = dl_manager.download_and_extract(URLS)
        label_dir = downloaded_files.pop("ureader_json")
        label_jsonl_ls = list(Path(label_dir).rglob("*.jsonl"))

        train_label = matching_path("train")
        valid_label = matching_path("val")
        test_label = matching_path("test")

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "filepath": train_label,
                    "split": "train",
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "filepath": valid_label,
                    "split": "validation",
                },
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "filepath": test_label,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: List[dict], split: str):
        def load_image_file(example):
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
                image_ls, conversations, dataset_name = data_row

                loaded_image_ls = list()
                loaded_image_size_ls = list()
                for img_path in image_ls:
                    try:
                        with warnings.catch_warnings(record=True) as warn_ls:
                            img_path = Path(img_path)
                            if not img_path.exists():
                                print(f"{img_path}가 발생함. 해당 샘플은 skip함.")
                                semaphore.release()
                                return None, None, None, None

                            img_bytes = img_path.read_bytes()
                            pil_image = PIL_Image.open(BytesIO(img_bytes))
                            pil_image.verify()

                            if warn_ls:
                                # DecompressionBombWarning이나 파일 일부가 손상된 경우 warn이 발생함.
                                # 특히 DecompressionBombWarning는 이미지가 너무 커서 문제가 발생하는 경우임.
                                # 이미지 저장 및 업로드 시 문제가 발생하기 때문에 건너뜀.
                                for warn in warn_ls:
                                    print(f"{warn.message}가 발생함. 해당 샘플은 skip함.")
                                pil_image.close()
                                semaphore.release()
                                return None, None, None, None

                        if pil_image.format.lower() not in ["jpg", "jpeg", "png", "webp"]:
                            pil_image.close()
                            semaphore.release()
                            return None, None, None, None

                        pil_image.close()
                        loaded_image_ls.append(img_bytes)
                        loaded_image_size_ls.append(pil_image.width * pil_image.height)
                    except BaseException as e:  # noqa: F841
                        print(f"{e}가 발생함. 해당 샘플은 skip함.")
                        semaphore.release()
                        return None, None, None, None

                img_token_num = 0
                new_conversation_ls = list()
                for chat in conversations:
                    content = convert_mm_content(chat["value"], r"\<image\>")

                    img_token_num += len([chat for chat in content if chat["type"] == "image"])

                    chat = {
                        "role": chat["from"],
                        "content": json.dumps(content, ensure_ascii=False),
                    }
                    new_conversation_ls.append(chat)

                if img_token_num != len(loaded_image_ls):
                    semaphore.release()
                    return None, None, None, None

                semaphore.release()
                return (
                    loaded_image_ls[0],
                    new_conversation_ls,
                    dataset_name,
                    sum(loaded_image_size_ls),
                )

            def data_generator():
                for ureader_instruct_row in UReaderInstruct_zip:
                    semaphore.acquire()
                    yield ureader_instruct_row

            image_ls, conversations_ls, dataset_name_ls = (
                example["image"] if isinstance(example["image"], list) else [example["image"]],
                example["conversations"] if isinstance(example["conversations"], list) else [example["conversations"]],
                example["date_from"] if isinstance(example["date_from"], list) else [example["date_from"]],
            )

            finish_image_ls = list()
            finish_image_size_ls = list()
            finish_conversations_ls = list()
            finish_dataset_name_ls = list()

            semaphore = Semaphore(self.thread_num * 2)
            loader = data_generator()
            UReaderInstruct_zip = zip(image_ls, conversations_ls, dataset_name_ls)
            with ThreadPool(self.thread_num) as thread_pool:
                thead_iter = thread_pool.imap_unordered(downloader, loader)
                for img_ls, conversations, dataset_name, img_size in thead_iter:
                    if not img_ls:
                        continue

                    finish_image_ls.append(img_ls)
                    finish_conversations_ls.append(conversations)
                    finish_dataset_name_ls.append(dataset_name)
                    finish_image_size_ls.append(img_size)

            return {
                "image": finish_image_ls,
                "conversations": finish_conversations_ls,
                "dataset_name": finish_dataset_name_ls,
                "image_size": finish_image_size_ls,
            }

        datasets = Dataset.from_list(filepath)

        finish_shard_ls = list()
        for shard_idx in range(self.shard_num):
            shard_datasets = datasets.shard(num_shards=self.shard_num, index=shard_idx)
            shard_datasets = shard_datasets.map(
                load_image_file,
                num_proc=self.num_proc,
                batched=self.batched,
                batch_size=self.batch_size,
                load_from_cache_file=True,
                remove_columns=datasets.column_names,
                desc=f"UReaderInstruct-{shard_idx}/{self.shard_num}",
            )
            finish_shard_ls.append(shard_datasets)

        datasets = concatenate_datasets(finish_shard_ls)
        image_size_ls = get_length_grouped_indices(
            datasets["image_size"],
            batch_size=1,
            mega_batch_mult=DEFAULT_MAX_BATCH_SIZE,
        )

        idx_ = 0
        for idx in image_size_ls:
            data = datasets[idx]
            del data["image_size"]
            data["id"] = idx_
            yield (idx_, data)
            idx_ += 1
