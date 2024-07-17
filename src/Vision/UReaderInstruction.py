import json
import os
import sys
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
from PIL import Image as PIL_Image

from transformers.trainer_pt_utils import get_length_grouped_indices


URLS = {
    "ChartQA": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/ChartQA.tar?download=true",
    "DeepForm": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/DeepForm.tar?download=true",
    "DocVQA": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/DocVQA.tar?download=true",
    "InfographicsVQA": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/InfographicsVQA.tar?download=true",
    "KleisterCharity": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/KleisterCharity.tar?download=true",
    "TabFact": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/TabFact.tar?download=true",
    "TextCaps": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/TextCaps.tar?download=true",
    "TextVQA": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/TextVQA.tar?download=true",
    "VisualMRC": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/VisualMRC.tar?download=true",
    "WikiTableQuestions": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/WikiTableQuestions.tar?download=true",
    "benchmark_files": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/benchmark_files.zip?download=true",
    "ureader_json": "https://huggingface.co/datasets/Mizukiluke/ureader-instruction-1.0/resolve/main/ureader_json.zip?download=true",
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
        check_dict = dict()
        test = [os.path.getsize(x["image"][0]) for x in filepath]
        test = get_length_grouped_indices(test, 1, mega_batch_mult=1000)
        for idx, x in enumerate(test):
            x = filepath[x]
            image_path = str(x["image"][0])

            conversations = list()
            for chat in x["conversations"]:
                if chat["value"] == "<image>":
                    continue
                chat = {"role": chat["from"], "content": chat["value"]}
                conversations.append(chat)

            try:
                img = PIL_Image.open(image_path)
                PIL_Image._decompression_bomb_check(img.size)
            except:
                print("pass")
                continue

            image_bytes = Path(image_path).read_bytes()

            data = {
                "image": image_bytes,
                "conversations": conversations,
                "metadata": {"source": x["date_from"]},
            }

            yield (idx, data)
            check_dict[image_path] = conversations
