import io
import json
import os
from pathlib import Path
from tarfile import TarFile
from zipfile import ZipFile

import datasets
from datasets import Features, Image, Sequence, Value
from PIL import Image as PIL_Image


PIL_Image.MAX_IMAGE_PIXELS = 809549650

_HOMEPAGE = "https://www.docvqa.org/"


class DocVQA(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = Features(
            {
                "questionId": Value("int32"),
                "question": Value("string"),
                "question_types": [Value("string")],
                "image": Image(),
                "docId": Value("int32"),
                "ucsf_document_id": Value("string"),
                "ucsf_document_page_no": Value("string"),
                "answers": [Value("string")],
                "data_split": Value("string"),
            }
        )

        return datasets.DatasetInfo(
            homepage=_HOMEPAGE,
            features=features,
            supervised_keys=None,
            citation=None,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": "/root/SP-DocVQA/spdocvqa_qas/train_v1.0_withQT.json",
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": "/root/SP-DocVQA/spdocvqa_qas/val_v1.0_withQT.json",
                    "split": "valid",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        for data in json.loads(Path(filepath).read_text())["data"]:
            try:
                img_pth = data["image"]
                img_pth = img_pth.replace("documents", "/root/SP-DocVQA/spdocvqa_images")
                img_bytes = Path(img_pth).read_bytes()

                PIL_Image.open(io.BytesIO(img_bytes))
                data["image"] = img_bytes
            except:
                print(f"""{img_pth} was passed\n""")
                continue

            yield (data["questionId"], data)
