import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List

from datasets import (
    BuilderConfig,
    DatasetInfo,
    Features,
    GeneratorBasedBuilder,
    Image,
    Split,
    SplitGenerator,
    Value,
)
from natsort import natsorted

from transformers import set_seed


set_seed(42)


CITATION = """\
@misc{han2023chartllama,
      title={ChartLlama: A Multimodal LLM for Chart Understanding and Generation}, 
      author={Yucheng Han and Chi Zhang and Xin Chen and Xu Yang and Zhibin Wang and Gang Yu and Bin Fu and Hanwang Zhang},
      year={2023},
      eprint={2311.16483},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""
HOMEPAGE = "https://github.com/tingxueronghua/ChartLlama-code"
_DATANAME = "ChartLlamaDataset"
DESCRIPTION = """"# 데이터 구조
{
    "id": "ours_simplified_qa_37_0",
    "image": "ours/box_chart/png/box_chart_100examples_37.png",
    "model": "",
    "conversations": [
        {"from": "human", "value": "<image>\nWhat is the title of the chart?"},
        {"from": "gpt", "value": "Analysis of smartphone usage patterns"},
    ],
}
# 폴더 구조
0c9d2a0ecefa932f49ed69f650495c9d4e8b10a021e765ac0687638d27bbcb7e
 ┗ ours
 ┃ ┣ box_chart
 ┃ ┃ ┗ png
 ┃ ┃ ┃ ┣ box_chart_100examples_0.png
 ┃ ┣ candlestick_chart
 ┃ ┃ ┗ png
 ┃ ┃ ┃ ┣ candlestick_chart_100examples_0.png
 ┃ ┣ funnel_chart
 ┃ ┃ ┗ png
 ┃ ┃ ┃ ┣ funnel_chart_100examples_0.png
 ┃ ┣ gantt_chart
 ┃ ┃ ┗ png
 ┃ ┃ ┃ ┣ gantt_chart_100examples_0.png
 ┃ ┣ heatmap_chart
 ┃ ┃ ┗ png
 ┃ ┃ ┃ ┣ heatmap_chart_100examples_0.png
 ┃ ┣ polar_chart
 ┃ ┃ ┗ png
 ┃ ┃ ┃ ┣ polar_chart_100examples_10.png
 ┃ ┗ scatter_chart
 ┃ ┃ ┗ png
 ┃ ┃ ┃ ┣ scatter_chart_100examples_10.png
와 같이 구성되어 있음.
그 외의 라벨들은 전부 json으로 되어 있고
"""


URLS = {
    "box_chart": "https://huggingface.co/datasets/listen2you002/ChartLlama-Dataset/resolve/main/box_chart_100examples_simplified_qa.json",
    "candlestick_chart": "https://huggingface.co/datasets/listen2you002/ChartLlama-Dataset/resolve/main/candlestick_chart_100examples_simplified_qa.json",
    "funnel_chart": "https://huggingface.co/datasets/listen2you002/ChartLlama-Dataset/resolve/main/funnel_chart_100examples_simplified_qa.json",
    "gantt_chart": "https://huggingface.co/datasets/listen2you002/ChartLlama-Dataset/resolve/main/gantt_chart_100examples_simplified_qa.json",
    "heatmap_chart": "https://huggingface.co/datasets/listen2you002/ChartLlama-Dataset/resolve/main/heatmap_chart_100examples_simplified_qa.json",
    "ours": "https://huggingface.co/datasets/listen2you002/ChartLlama-Dataset/resolve/main/ours.zip",
    "polar_chart": "https://huggingface.co/datasets/listen2you002/ChartLlama-Dataset/resolve/main/polar_chart_100examples_simplified_qa.json",
    "scatter_chart": "https://huggingface.co/datasets/listen2you002/ChartLlama-Dataset/resolve/main/scatter_chart_100examples_simplified_qa.json",
}


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


class ChartLlamaDataset(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="SINGLE-SFT",
            data_dir="SINGLE-SFT",
            version="1.0.0",
            description="""GPT-4를 활용하여 고품질의 명령어 튜닝 데이터 세트를 생성한 데이터.
유연성 덕분에 낮은 리소스 지출을 유지하면서 다양하고 고품질의 인스트럭션 튜닝 데이터를 일관되고 효율적으로 생성함. 
표 형식의 데이터 생성, 차트 수치 생성, 인스트럭션 튜닝 데이터 설계를 각각 다른 단계에서 담당하는 다단계 데이터 생성 프로세스를 개발.
원본 데이터 그대로\n"""
            + DESCRIPTION,
        ),
        BuilderConfig(
            name="MULTI-SFT",
            data_dir="MULTI-SFT",
            version="1.0.0",
            description="""GPT-4를 활용하여 고품질의 명령어 튜닝 데이터 세트를 생성한 데이터.
유연성 덕분에 낮은 리소스 지출을 유지하면서 다양하고 고품질의 인스트럭션 튜닝 데이터를 일관되고 효율적으로 생성함. 
표 형식의 데이터 생성, 차트 수치 생성, 인스트럭션 튜닝 데이터 설계를 각각 다른 단계에서 담당하는 다단계 데이터 생성 프로세스를 개발.
데이터가 한 이미지에 여러 대화를 이어놓은 형태다 보니 한 이미지로 멀티턴 대화 데이터를 만듬.\n"""
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

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download(URLS)
        image_files = dl_manager.extract(downloaded_files.pop("ours"))

        img_path_dict = {path.stem: path for path in Path(image_files).glob("*/*")}
        lbl_file_dict = {task_name: Path(json_path) for task_name, json_path in downloaded_files.items()}

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "img_path_dict": img_path_dict,
                    "lbl_file_dict": lbl_file_dict,
                },
            ),
        ]

    def _generate_examples(self, img_path_dict: Dict[str, Path], lbl_file_dict: Dict[str, Path]) -> Generator:
        if self.config.name == "SINGLE-SFT":
            generator = self._generate_examples_single_sft(img_path_dict, lbl_file_dict)
        elif self.config.name == "MULTI-SFT":
            generator = self._generate_examples_multi_sft(img_path_dict, lbl_file_dict)

        for idx, data in enumerate(generator):
            yield idx, data

    def _generate_examples_single_sft(
        self,
        img_path_dict: Dict[str, Path],
        lbl_file_dict: Dict[str, Path],
    ) -> Generator:
        def get_valid_conversations(conversations: List[Dict[str, str]]):
            if len(conversations) != 2:
                return None

            user, assistant = conversations

            if user["from"] != "human":
                return None

            if assistant["from"] != "gpt":
                return None

            user_content = convert_mm_content(user["value"], "<image>")
            assistant_content = convert_mm_content(assistant["value"], "<image>")

            if user_content == [] or assistant_content == []:
                return None

            return [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]

        for task, file_path in lbl_file_dict.items():
            label_ls = json.loads(file_path.read_text())
            img_path_map = {path.name: path for path in img_path_dict[task].rglob("*") if path.is_file()}
            for label in label_ls:
                image = img_path_map[Path(label["image"]).name].read_bytes()

                conversations = get_valid_conversations(label["conversations"])
                if conversations is None:
                    breakpoint()
                    print(f"Invalid conversation: [{label['id']}]{label['image']}")
                    continue

                yield {
                    "id": label["id"],
                    "image": image,
                    "conversations": conversations,
                }

    def _generate_examples_multi_sft(
        self,
        img_path_dict: Dict[str, Path],
        lbl_file_dict: Dict[str, Path],
    ) -> Generator:
        def get_valid_conversations(multi_label_ls):
            new_conversations = list()
            for label in multi_label_ls:
                if len(label["conversations"]) != 2:
                    continue

                user, assistant = label["conversations"]

                if user["from"] != "human":
                    continue

                if assistant["from"] != "gpt":
                    continue
                user_content = convert_mm_content(user["value"], "<image>")
                assistant_content = convert_mm_content(assistant["value"], "<image>")

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

        for task, file_path in lbl_file_dict.items():
            label_ls = json.loads(file_path.read_text())
            label_ls = natsorted(label_ls, key=lambda x: x["id"])
            img_path_map = {path.name: path for path in img_path_dict[task].rglob("*") if path.is_file()}
            multi_turn_dict = defaultdict(list)
            for label in label_ls:
                multi_turn_dict[Path(label["image"]).name].append(label)

            for img_name, multi_label_ls in multi_turn_dict.items():
                image = img_path_map[img_name].read_bytes()

                conversations = get_valid_conversations(multi_label_ls)
                if conversations == []:
                    breakpoint()
                    print(f"Invalid conversation: {img_name}")
                    continue

                yield {
                    "id": img_name.replace(".png", ""),
                    "image": image,
                    "conversations": conversations,
                }
