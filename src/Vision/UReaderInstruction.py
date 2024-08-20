import json
import os
import warnings
from pathlib import Path
from typing import List

from datasets import (
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


class UReaderInstruction(GeneratorBasedBuilder):
    VERSION = Version("1.1.0")

    def _info(self):
        return DatasetInfo(
            features=Features(
                {
                    "image": Image(),
                    "conversations": [{"role": Value("string"), "content": Value("string")}],
                    "metadata": {"source": Value("string")},
                }
            ),
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
                        img_ls.append(Path(data_dir, img))
                    line["image"] = img_ls
                    line["date_from"] = start_name
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
        img_size_ls = [os.path.getsize(x["image"][0]) for x in filepath]
        img_size_ls = get_length_grouped_indices(
            img_size_ls,
            batch_size=1,
            mega_batch_mult=DEFAULT_MAX_BATCH_SIZE,
        )

        for idx, data_idx in enumerate(img_size_ls):
            dataset = filepath[data_idx]

            if len(dataset["image"]) >= 2:
                warnings.warn("해당 데이터의 이미지는 두개임.")
                continue

            img_path = dataset["image"][0]
            conversations = dataset["conversations"]

            # 모든 데이터가 싱글턴이라, 이렇게 만들어도 됨.
            user_conversations = [chat for chat in conversations if chat["from"] == "user"]
            assistant_conversations = [
                {
                    "role": "assistant",
                    "content": json.dumps([{"content": "text", "text": chat["value"]}], ensure_ascii=False),
                }
                for chat in conversations
                if chat["from"] == "assistant"
            ]

            new_contents = list()
            for chat in user_conversations:
                content = {"type": "image"} if chat["value"] == "<image>" else {"type": "text", "text": chat["value"]}
                new_contents.append(content)

            new_conversations = [{"role": "user", "content": json.dumps(new_contents, ensure_ascii=False)}]
            new_conversations.extend(assistant_conversations)

            data = {
                "image": img_path.read_bytes(),
                "conversations": new_conversations,
                "metadata": {"source": dataset["date_from"]},
            }

            yield (idx, data)
