import json
import os
from pathlib import Path
from time import sleep
from typing import Generator, List

import tiktoken
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
    load_dataset,
    load_from_disk,
)
from langdetect import detect_langs
from openai import OpenAI


GPT_VERSION = "gpt-4o-mini-2024-07-18"

_HOMEPAGE = "https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-118K"


_DATANAME = "KoLLaVAReCap118K"


_DESCRIPTION = f"{GPT_VERSION}를 사용해 lmms-lab/LLaVA-ReCap-118K를 한국어로 번역한 데이터"

# copied from: https://community.openai.com/t/i-need-your-help-with-prompt/860497/7
SYSTEM = """You are an English Translator, specialized in translating sentence from English into korea. Your main goals are to ensure grammatically correct translations and deliver text that feels natural and human-oriented. 

Instructions:
1. Translate the provided sentence from English to the korea.
2. Ensure that the translation maintains the meaning and context of the original text.
3. Use appropriate grammar, syntax, and idiomatic expressions to make the translation sound natural.
4. Avoid literal translations unless necessary to preserve the meaning.
5. If there are cultural references or idioms, adapt them to be understandable and relevant in the korea.
6. Keep the formatting and structure of the original text intact unless specified otherwise.
7. Review the translation for any errors or awkward phrasing before finalizing."""

client = OpenAI()


def num_tokens_from_messages(messages, model=GPT_VERSION):
    "Return the number of tokens used by a list of messages."
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")

    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        print("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")
        return num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        print("Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")
        return num_tokens_from_messages(messages, model="gpt-4o-2024-08-06")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(f"num_tokens_from_messages() is not implemented for model {model}.")

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


class KoLLaVAReCap118K(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(name="chat", version="1.0.0", description=_DESCRIPTION),
    ]
    DEFAULT_CONFIG_NAME = "chat"

    def _info(self):
        features = Features(
            {
                "id": Value("string"),
                "image": Image(),
                "conversations": [{"role": Value("string"), "content": Value("string")}],
                "data_source": Value("string"),
                "caption": Value("string"),
                "en_caption": Value("string"),
                "caption_ls": [Value("string")],
            }
        )

        self.shard_num = int(os.getenv("SHARD_NUM", "10"))
        self.gpt_version = os.getenv("GPT_VERSION", GPT_VERSION)
        self.map_batch_size = int(os.getenv("MAP_BATCH_SIZE", "200"))
        self.map_num_proc = int(os.getenv("MAP_NUM_PROC", "4"))

        self.filter_min_len = int(os.getenv("FILTER_MIN_LEN", "40"))
        self.filter_max_len = int(os.getenv("FILTER_MAX_LEN", "1000"))

        return DatasetInfo(
            description=self.config.description,
            version=self.config.version,
            features=features,
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager) -> List[SplitGenerator]:  # type: ignore
        def filter_function(conversations) -> List[bool]:
            filter_flag_ls = list()
            for conversation in conversations:
                try:
                    content = conversation[-1]["value"]
                    message = [{"role": "user", "content": content}]

                    if not content:
                        filter_flag_ls.append(False)
                        continue

                    detect_result = detect_langs(content)[0]
                    if detect_result.lang != "en" or detect_result.prob <= 0.8:
                        filter_flag_ls.append(False)
                        continue

                    token_num = num_tokens_from_messages(message, self.gpt_version)

                    if not (self.filter_min_len <= token_num <= self.filter_max_len):
                        filter_flag_ls.append(False)
                        continue

                    filter_flag_ls.append(True)
                except BaseException:
                    filter_flag_ls.append(False)

            return filter_flag_ls

        cache_dir = Path(dl_manager.download_config.cache_dir, _DATANAME)

        dataset = load_dataset("lmms-lab/LLaVA-ReCap-118K")
        dataset = dataset.filter(
            filter_function,
            input_columns=["conversations"],
            num_proc=6,
            batched=True,
            batch_size=1000,
        )

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "dataset": dataset["train"],
                    "split": "train",
                    "cache_dir": cache_dir,
                },
            ),
        ]

    def _generate_examples(self, dataset: Dataset, split: str, cache_dir: Path) -> Generator:
        def translate_en_to_ko(example):
            def send_request_to_gpt(
                message: List[dict],
                model: str,
                seed: int = 42,
                retry: int = 10,
                error_interval_time: int = 10,
            ) -> str:
                for retries in range(retry):
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=message,
                            response_format={"type": "text"},
                            seed=seed,
                        )
                        augmentate_answer = response.choices[0].message.content

                        break
                    except BaseException as e:
                        print(f"{retries}회의 instruction 생성 중 {e}과 같은 애러가 발생해 다시 retry함.")
                        sleep(error_interval_time)
                else:
                    augmentate_answer = ""

                return augmentate_answer

            check_input_token_ls, check_output_token_ls = list(), list()
            finish_conversations_ls, finish_en_ls, finish_ko_ls = list(), list(), list()
            for conversations in example["conversations"]:
                new_conversations = list()
                for chat in conversations:
                    match chat["from"]:
                        case "human":
                            new_content = json.dumps([{"type": "image"}], ensure_ascii=False)
                            new_conversations.append({"role": "user", "content": new_content})
                        case "gpt":
                            input_message = [
                                {"role": "system", "content": SYSTEM},
                                {"role": "user", "content": f"{chat['value']} to 한국어"},
                            ]

                            ko_recaption = send_request_to_gpt(input_message, gpt_version)

                            output_message = [{"role": "assistant", "content": ko_recaption}]

                            new_content = json.dumps([{"type": "text", "text": ko_recaption}], ensure_ascii=False)
                            new_conversations.append({"role": "assistant", "content": new_content})

                            # NOTE: conversation 애서 assistant가 1개인 상황에서 정상적으로 동작함.
                            finish_ko_ls.append(ko_recaption)
                            finish_en_ls.append(chat["value"])

                            check_input_token_ls.append(num_tokens_from_messages(input_message))
                            check_output_token_ls.append(num_tokens_from_messages(output_message))

                finish_conversations_ls.append(new_conversations)

            example["conversations"] = finish_conversations_ls

            example["caption"] = finish_ko_ls
            example["en_caption"] = finish_en_ls

            example["input_token_num"] = check_input_token_ls
            example["output_token_num"] = check_output_token_ls

            return example

        digits = len(str(self.shard_num))
        cache_path = cache_dir.joinpath(split)

        gpt_version = self.gpt_version

        finish_shard_ls = list()
        for shard_idx in range(self.shard_num):
            dir_name = f"[{gpt_version}]{_DATANAME}-{self.shard_num}/{shard_idx + 1:0{digits}d}"
            cache_file_name = cache_path.joinpath(dir_name)

            if cache_file_name.exists():
                shard_dataset = load_from_disk(cache_file_name.as_posix())
                finish_shard_ls.append(shard_dataset)
                continue

            shard_dataset: Dataset = dataset.shard(self.shard_num, shard_idx)
            shard_dataset = shard_dataset.map(
                translate_en_to_ko,
                num_proc=self.map_num_proc,
                batch_size=self.map_batch_size,
                batched=True,
                desc=dir_name,
            )

            shard_dataset.save_to_disk(cache_file_name)
            finish_shard_ls.append(shard_dataset)

        dataset = concatenate_datasets(finish_shard_ls)
        for idx, data in enumerate(dataset):
            data["caption_ls"] = [data["caption"]]
            yield (idx, data)


# simple EDA and data check
"""
from tqdm import tqdm
import tiktoken
from collections import Counter
from langdetect import detect, detect_langs

conversations = [
    chat["value"]
    for conversation in tqdm(dataset["conversations"])
    for chat in conversation
    if chat["from"] == "gpt"
]
num_token_ls = [
    num_tokens_from_messages(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{en_sentence} to 한국어"},
        ],
        model=self.gpt_version,
    )
    for en_sentence in tqdm(conversations)
]
num_token_ls_only_input = [
    num_tokens_from_messages(
        [
            {"role": "user", "content": en_sentence},
        ],
        model=self.gpt_version,
    )
    for en_sentence in tqdm(conversations)
]

token_ls = [
    (
        en_sentence,
        num_tokens_from_messages(
        [
            {"role": "user", "content": en_sentence},
        ],
        model=self.gpt_version,
    ))
    for en_sentence in tqdm(conversations)
]
token_ls = sorted(token_ls, key=lambda x: x[-1])

histogram = list(Counter([token[-1] for token in token_ls]).items())
sorted(histogram, key=lambda x: x[0])

lang_detect_check_ls = list()
for token in tqdm(token_ls):
    try:
        lang_detect_check_ls.append(detect_langs(token[0]))
    except BaseException:
        continue
lang_flatten_ls = sum(lang_detect_check_ls, [])
Counter([detected.lang for detected in lang_flatten_ls])
{'en': 118242, 'zh-cn': 55, 'fr': 4, 'cy': 4, 'es': 3, 'de': 3, 'it': 3, 'vi': 3, 'ca': 2, 'fi': 2, 'no': 2, 'ta': 1, 'hu': 1, 'id': 1, 'lv': 1, 'sk': 1, 'uk': 1, 'ro': 1, 'sv': 1, 'ko': 1, 'pt': 1, 'af': 1}

'The image depicts an urban street scene with a focus on a bus stop sign. The sign is blue with white text that reads "Bus Stop" at the top. Below this, there are several lines of text indicating bus routes and numbers, such as "1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1A 1B 1C 1'

".................................................................................................................................."
"The image显示了一个常规的、重复的、无意义的字符串。"


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

data_ls = Path("").glob("*")

_data_ls = list()
input_token_num = list()
output_token_num = list()
for disk_path in data_ls:
    data = load_from_disk(disk_path.as_posix())

    input_token_num.append(sum(data["input_token_num"]))
    output_token_num.append(sum(data["output_token_num"]))
    _data_ls.append(data)


data = concatenate_datasets(_data_ls)

caption_ls = data["caption"]
lang_ls = [detect_langs(x)[0] for x in tqdm(caption_ls)]

Counter(x.lang for x in lang_ls)
# Counter({'ko': 118072, 'id': 1})

[idx for idx, x in enumerate(lang_ls) if x.lang == "id"]
# [54673]

data[54673]["caption"]
# '이미지에는 바닥에 놓인 두 개의 빈티지 여행 가방이 보입니다. 왼쪽의 가방은 더 크고 직사각형이며, 노란 갈색 외관과 검은 손잡이가 있습니다. 가죽 또는 가죽과 유사한 재질로 만들어진 것으로 보이며, 앞쪽에 금속 걸쇠가 있습니다. 오른쪽의 가방은 더 작고 나무로 만들어졌으며, 측면에 파란색과 흰색의 표지가 부착되어 있습니다. 표지에는 "KENNEDY\'S SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT"라고 적혀 있습니다.'
data[54673]["en_caption"]
# 'The image shows two vintage suitcases placed on a floor. The suitcase on the left is a larger, rectangular piece with a yellowish-brown exterior and a black handle. It appears to be made of leather or a leather-like material, and there is a metal latch on the front. The suitcase on the right is smaller, with a wooden construction and a blue and white sign attached to its side. The sign reads "KENNEDY\'S SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SCUIT SC'

data[54673]["image"].save("outlier.png")
# 실제 이미지에 저런 문자열 없음 ㅋㅋㅋㅋ 이상한 데이터임.

[idx for idx, x in enumerate(lang_ls) if x.prob <= 0.7]
# [30971, 32054, 78196]

data[30971]["caption"]
# '이미지는 횡단보도에 설치된 방향 표지판들이 붙어 있는 기둥을 보여줍니다. 표지판은 프랑스어로 되어 있으며, 다양한 방향을 가리키고 있어 거리 이름이나 위치를 나타냅니다. 위에서 아래로 표지판에는 다음과 같이 적혀 있습니다.\n\n1. "Bureau de Poste 5, Place des Abbesses"\n2. "Butte Montmartre"\n3. "Musee de la Vie Romantique 16, Rue Chaptal"\n4. "Theatre FONTAINE 9, Rue Fontaine"\n5. "Maison des Associations du 18e"\n6. "Place Clichy"\n7. "Place Stalingrad"\n\n표지판은 직사각형 형태로, 파란 배경에 하얀 글씨로 쓰여 있습니다. 첫 번째 표지판만 흰색 배경에 검은 글씨로 되어 있습니다. 표지판이 설치된 기둥은 거리의 모퉁이에 있으며, 배경에는 건물들이 있어 도시 환경을 암시합니다. 하늘은 흐리고 전체 조명은 부드러워서 이른 아침이나 늦은 오후로 보입니다.'

data[32054]["caption"]
# '이미지에서 한 남자가 베이지색 벽이 있는 방에 앉아 있습니다. 그는 흰색 셔츠를 입고 다양한 스티커가 붙어 있는 노트북을 들고 있습니다. 스티커들은 다채로운 색상으로, "drop.io", "Revision3", "Mashable", "NY", "Contagious", "Web 2.0", "Social Media Club", "Make Good", "Nothing", "Mashable\'s Leaders", "Alex", "37signals", "Bricabox", "Cream", "AIM"과 같은 텍스트와 그래픽이 혼합되어 있습니다. 남자는 미소를 지으며 기분이 좋은 모습입니다. 노트북은 열려 있고 전원에 연결되어 있어 그는 작업을 하거나 인터넷을 탐색하고 있을 가능성이 있습니다. 전체적인 장면은 캐주얼하고 편안한 분위기를 전달합니다.'

data[78196]["caption"]
# '이미지에는 대형 관광버스가 포장된 표면에 주차되어 있고 배경에는 푸른 식물과 나무가 있습니다. 버스는 커다란 기타 이미지와 "Gibson"이라는 문구가 돋보이는 생동감 넘치는 그래픽 디자인으로 장식되어 있습니다. 기타 아래에는 "The Gibson Family of Brands"라는 추가 문구가 있으며, 그 뒤에는 "Baldwin," "Chicken Pickin\'," "D\'Addario," "Echoplex," "Electric Guitar," "Epiphone," "Flatiron," "Gibson," "Gibson Guitar Education," "Hamilton," "Kramer," "Maestro," 그리고 "Mitchell"과 같은 브랜드 이름 목록이 이어집니다. 버스에는 등록 번호판도 보이지만, 글씨가 너무 작아 명확하게 읽을 수 없습니다. 전체 이미지 스타일은 버스와 그 브랜드에 초점을 맞춘 실사 photograph입니다.'

[idx for idx, x in enumerate(lang_ls) if x.prob <= 0.9]
# [30971, 32054, 67852, 78196, 90866]

data[67852]["caption"]
# '제공하신 이미지는 빈 이미지 파일이거나 손상된 것으로 보입니다. 구별 가능한 특징이나 내용 없이 균일한 색상만이 나타나고 있습니다. 여기 supposed to be an image if true, but it is not displaying correctly. 이미지를 설명할 수 있도록 유효한 이미지 파일을 제공해 주세요.'

data[90866]["caption"]
# '이미지는 "The Red Seal"이라는 빈티지 잡지의 표지입니다. 표지에는 도시 풍경 위를 비행하는 작은 프로펠러 비행기의 흑백 사진이 특징적으로 담겨 있습니다. 이 비행기는 단일 엔진의 고익 항공기로, 등록 번호는 N50000입니다. 표지의 텍스트는 이 잡지가 1963년에 발행되었으며, 15권 3호라고 명시하고 있습니다. 표지에는 Continental Motors Corporation, Continental Aviation & Engineering Corporation, Gray Marine Motor Company를 포함하는 Continental Family of Power Specialists에서 발행했다고도 언급되어 있습니다. 표지에는 또한 "여기, 캔자스 주 위치타에 있는 고향 위를 비행 중인 Cessna Aircraft Company가 제작한 100번째 항공기입니다. 총 100대 중 대다수와 최근 몇 년 동안 제작된 Cessna 100%가 다수와 함께, No. 50000은 Continental 엔진으로 비행합니다."라는 작은 텍스트가 포함되어 있습니다. 이는 이 비행기가 Continental 엔진으로 구동된다는 것을 시사합니다. 전체적인 이미지 스타일은 20세기 중반 항공 잡지를 연상케 하며, 항공기와 그를 만든 회사에 초점을 맞추고 있습니다.'

[idx for idx, x in enumerate(caption_ls) if "빈 이미지 파일" in x]
[57787, 67852]

data[57787]["caption"]
# '제공하신 이미지는 빈 이미지 파일이거나 손상된 것처럼 보입니다. 일관된 이미지를 형성하지 않는 반복적인 패턴이나 문자 시리즈가 표시됩니다. 해당 내용은 사진이나 일러스트레이션과 같은 표준 이미지 형식으로 인식할 수 없습니다. 특정 이미지를 보여주려 하셨다면, 다른 파일을 제공하시거나 의도한 내용을 명확히 해주시기 바랍니다.'

data[67852]["caption"]
# '제공하신 이미지는 빈 이미지 파일이거나 손상된 것으로 보입니다. 구별 가능한 특징이나 내용 없이 균일한 색상만이 나타나고 있습니다. 여기 supposed to be an image if true, but it is not displaying correctly. 이미지를 설명할 수 있도록 유효한 이미지 파일을 제공해 주세요.'

[x for idx, x in enumerate(caption_ls) if "손상된 것" in x]
# [
#     "제공하신 이미지는 저해상도이거나 손상된 것처럼 보입니다. 이로 인해 특정 세부사항을 파악하기 어렵습니다. 색상과 형태가 뒤섞여 coherent한 이미지가 형성되지 않는 것 같습니다. 다른 이미지가 있거나 다른 도움이 필요하시면 언제든지 말씀해 주세요!",
#     "한 30개 정도 있음...............................................................................................................",
#     "제공하신 이미지는 빈 이미지이거나 손상된 것으로 보입니다. 구별할 수 있는 특징이나 내용 없이 균일한 색상만 나타나고 있습니다. 여기 보여져야 할 이미지가 있다면 제대로 표시되지 않고 있습니다. 다른 이미지를 제공해 주시면 설명드리겠습니다.",
# ]

sorted([x for idx, x in enumerate(caption_ls) if "저해상도" in x])
# [
#     "귀하가 제공한 이미지는 꽤 작고 세부정보가 부족하여 특정 기능을 파악하기 어렵습니다. 픽셀화가 심한 저해상도 이미지로 보이며, 이는 썸네일이나 매우 작은 이미지일 가능성을 시사합니다. 더 명확한 이미지나 추가적인 정보를 제공해 주시면, 기꺼이 이를 설명하는 데 도움을 드리겠습니다.",
#     "..................... 한 200개 정도 있음.",
#     "제공하신 이미지는 해변 풍경의 저해상도 흐릿한 사진인 것 같습니다. 명확성이 부족해 특정 세부 사항을 구분하기 어렵지만, 몇 가지 일반적인 요소는 확인할 수 있습니다. 모래가 있는 지역이 있는 것으로 보이며, 이는 해변이나 해안선을 나타낼 가능성이 있습니다. 식물이나 풀 같은 것도 있을 수 있지만, 확실히 확인하기는 어렵습니다. 조명 상태로 볼 때 낮인 것처럼 보이지만, 선명도가 부족해 정확한 시간대를 결정하는 데 어려움이 있습니다. 전반적인 인상은 자연의 야외 환경으로, 아마도 해안 지역일 가능성이 높지만, 더 자세한 정보가 없이는 더 정확한 설명을 제공할 수 없습니다.",
# ]
"""
