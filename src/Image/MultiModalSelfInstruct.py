import re
from collections import defaultdict
from typing import Generator

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
    load_dataset,
)


HOMEPAGE = "https://multi-modal-self-instruct.github.io/"

CITATION = """@article{zhang2024multimodal,
  title={Multimodal Self-Instruct: Synthetic Abstract Image and Visual Reasoning Instruction Using Language Model},
  author={Zhang, Wenqi and Cheng, Zhenglin and He, Yuanyu and Wang, Mengna and Shen, Yongliang and Tan, Zeqi and Hou, Guiyang and He, Mingqian and Ma, Yanna and Lu, Weiming and others},
  journal={arXiv preprint arXiv:2407.07053},
  year={2024}
}"""


DESCRIPTION = """"# 데이터 구조
{
    "question_id": "20231127220121117781",
    "question": "<image>\nWhat is the lowest mortality rate recorded and which country and year does it correspond to?",
    "image_path": "chart/20231127220121117781.png",
    "image": <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=1419x926 at 0x7FADC6B829E0>,
    "answer": "The answer is The lowest mortality rate recorded is 6.81 for Australia in 2010..",
}
얘들 한 이미지 마다 여러 질문 답변 쌍의 데이터로 만들어 놨는데
question_id가 이미지 이름으로 만들어 졌다 보니 중복되는 녀석들이 존재함."""


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


class MultiModalSelfInstruct(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="SINGLE-SFT",
            data_dir="SINGLE-SFT",
            version="1.0.0",
            description="""차트, 표, 시뮬레이션 지도, 대시보드, 순서도, 관계 그래프, 평면도, 시각적 퍼즐 등 8가지 시각적 시나리오에 대한 11,193개의 지침이 포함된 데이터로,
차트, 지도, 레이아웃과 같은 추상적인 이미지에 대한 이해와 시각적 추론 능력부터 시계에서 시간을 읽거나 순서도를 이해하거나
로드맵을 사용하여 경로를 계획하는 것과 같은 간단한 일상 업무 대해 이미지와 시각적 추론 지침을 포함한 합성\n"""
            + DESCRIPTION,
        ),
        BuilderConfig(
            name="MULTI-SFT",
            data_dir="MULTI-SFT",
            version="1.0.0",
            description="""차트, 표, 시뮬레이션 지도, 대시보드, 순서도, 관계 그래프, 평면도, 시각적 퍼즐 등 8가지 시각적 시나리오에 대한 11,193개의 지침이 포함된 데이터로,
차트, 지도, 레이아웃과 같은 추상적인 이미지에 대한 이해와 시각적 추론 능력부터 시계에서 시간을 읽거나 순서도를 이해하거나
로드맵을 사용하여 경로를 계획하는 것과 같은 간단한 일상 업무 대해 이미지와 시각적 추론 지침을 포함한 합성\n"""
            + DESCRIPTION,
        ),
    ]
    DEFAULT_CONFIG_NAME = "SINGLE-SFT"

    def _info(self):
        features = Features(
            {
                "id": Value("string"),
                "image": Image(),
                "conversations": [{"role": Value("string"), "content": Value("string")}],
            }
        )

        return DatasetInfo(
            description=self.config.description,
            version=self.config.version,
            features=features,
            homepage=HOMEPAGE,
            citation=CITATION,
        )

    def _split_generators(self, _):
        dataset = load_dataset(
            "zwq2018/Multi-modal-Self-instruct",
            features=Features(
                {
                    "question_id": Value(dtype="string", id=None),
                    "question": Value(dtype="string", id=None),
                    "image_path": Value(dtype="string", id=None),
                    "image": Image(mode=None, decode=False, id=None),
                    "answer": Value(dtype="string", id=None),
                }
            ),
        )

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "dataset": dataset["train"],
                },
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "dataset": dataset["test"],
                },
            ),
        ]

    def _generate_examples(self, dataset: Dataset) -> Generator:
        if self.config.name == "SINGLE-SFT":
            generator = self._generate_examples_single_sft(dataset)
        elif self.config.name == "MULTI-SFT":
            generator = self._generate_examples_multi_sft(dataset)

        for idx, data in enumerate(generator):
            data["id"] = str(idx)
            yield idx, data

    def _generate_examples_single_sft(self, dataset: Dataset) -> Generator:
        for data in dataset:
            conversations = [
                {"role": "user", "content": convert_mm_content(data["question"], "<image>")},
                {"role": "assistant", "content": convert_mm_content(data["answer"], "<image>")},
            ]

            if any(conv["content"] == [] for conv in conversations):
                print(f"Invalid conversation: {data['question_id']}")
                continue

            yield {
                "image": data["image"]["bytes"],
                "conversations": conversations,
            }

    def _generate_examples_multi_sft(self, dataset: Dataset) -> Generator:
        def get_valid_conversations(multi_label_ls):
            new_conversations = list()
            for label in multi_label_ls:
                user, assistant = label["question"], label["answer"]

                user_content = convert_mm_content(user, "<image>")
                assistant_content = convert_mm_content(assistant, "<image>")

                if user_content == [] or assistant_content == []:
                    continue

                if new_conversations != []:
                    # 이미 값이 차 있다는건 한번 턴이 진행 되었다는 거.
                    user_content = [content for content in user_content if content["type"] != "image"]

                new_conversations.extend(
                    [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ]
                )

            return new_conversations

        multi_turn_dict = defaultdict(list)
        for data_idx, question_id in enumerate(dataset["question_id"]):
            multi_turn_dict[question_id].append(data_idx)
        for _, id_ls in multi_turn_dict.items():
            selected_dataset = dataset.select(id_ls)
            conversations = get_valid_conversations(selected_dataset.select_columns(["question", "answer"]))

            yield {
                "image": selected_dataset[0]["image"]["bytes"],
                "conversations": conversations,
            }
