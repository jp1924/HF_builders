import io
import json
from pathlib import Path

import datasets
from datasets import Features, Image, Value
from natsort import natsorted
from PIL import Image as PIL_Image

_DESCRIPTION = """Visual question answering generally aims to answer a query described in natural language, taking cues from the document image as the only input. As a part of this competition, we propose a visual question answering a dataset and baseline model from business document images. While a lot of work has already been done in the broader of this space, the questions from business documents present many niche challenges that may require cross-document referencing, additional numeric computations over the simple search query to reach the final solution, and so on. Further, since most business documents are usually presented in a tabular format, it may be non-trivial to leverage this structural conformity to answer more challenging queries. Given the unique nature of the problem, its tremendous prospect in the industry, layers of challenges to be tackled, and the recent surge of interest in the broader space of visual question answering, we believe this problem would interest the research community worldwide and attract good participation."""

_URLs = {
    "train": "https://ilocr.iiit.ac.in/vqabd/assets/dataset/competition.zip",
    "validation": "https://ilocr.iiit.ac.in/vqabd/assets/dataset/competition_val.zip",
    "test": "https://ilocr.iiit.ac.in/vqabd/assets/dataset/VQAonBD_testset.zip",
}


class VQAonBD(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = Features(
            {
                "table_structure": [
                    {
                        "cell_id": Value("int32"),
                        "bbox": [Value("int32")],
                        "start_row": Value("int32"),
                        "start_col": Value("int32"),
                        "end_row": Value("int32"),
                        "end_col": Value("int32"),
                        "content": Value("string"),
                    }
                ],
                "questions_answers": [
                    {
                        "question_id": Value("string"),
                        "question": Value("string"),
                        "answer": Value("string"),
                        "answer_type": Value("string"),
                    }
                ],
                "file_name": Value("string"),
                "image": Image(),
            }
        )

        return datasets.DatasetInfo(
            features=features,
            supervised_keys=None,
            citation=None,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLs)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": downloaded_files["validation"],
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        if "train" == split:
            source_ls = list(Path(filepath, "competition", "table_images").glob("*.png"))
            label_ls = list(Path(filepath, "competition", "ground_truth").glob("*.json"))
        if "validation" == split:
            source_ls = list(Path(filepath, "competition", "table_images").glob("*.png"))
            label_ls = list(Path(filepath, "competition", "ground_truth").glob("*.json"))

        source_ls = natsorted(source_ls)
        label_ls = natsorted(label_ls)

        for idx, (src_file, label_file) in enumerate(zip(source_ls, label_ls)):
            image = src_file.read_bytes()
            label = json.loads(Path(label_file).read_text())

            table_structure_ls = list()
            for cell_id, cell in label["table_structure"].items():
                cell["cell_id"] = cell_id
                table_structure_ls.append(cell)

            questions_answers_ls = list()
            for category, label_dict in label["questions_answers"].items():
                for _id, _label in label_dict.items():
                    _label["question_id"] = f"{category}-{_id}"
                    _label["answer"] = str(_label["answer"])
                    questions_answers_ls.append(_label)

            label["table_structure"] = table_structure_ls
            label["questions_answers"] = questions_answers_ls
            label["image"] = image
            label["file_name"] = label_file.stem

            yield (idx, label)
