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


client = OpenAI()

_VERSION = Version("1.0.0")
GPT_VERSION = "gpt-4o-mini-2024-07-18"

_DESCRIPTION = f"{GPT_VERSION}를 사용해 lmms-lab/LLaVA-ReCap-118K를 한국어로 번역한 데이터"
_DATANAME = "KoLLaVAReCap118K"

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
        BuilderConfig(name="chat", version=_VERSION, description=_DESCRIPTION),
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

        self.shard_num = os.getenv("SHARD_NUM", 10)
        self.gpt_version = os.getenv("GPT_VERSION", GPT_VERSION)
        self.map_batch_size = os.getenv("MAP_BATCH_SIZE", 200)
        self.map_num_proc = os.getenv("MAP_NUM_PROC", 4)

        self.filter_min_len = os.getenv("FILTER_MIN_LEN", 40)
        self.filter_max_len = os.getenv("FILTER_MAX_LEN", 1000)

        return DatasetInfo(
            features=features,
            version=_VERSION,
            description=_DESCRIPTION,
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
        print(dataset)
        dataset = dataset.filter(
            filter_function,
            input_columns=["conversations"],
            num_proc=6,
            batched=True,
            batch_size=1000,
        )
        print(dataset)
        # dataset = dataset.sort(column_names="id")

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

                        # NOTE: 이건 전부 출력한 뒤 하는게 좋을 듯.
                        # detect_result = detect_langs(augmentate_answer)[0]
                        # if detect_result.lang != "ko" or detect_result.prob <= 0.8:
                        #     print(message, augmentate_answer)
                        #     continue

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
            dir_name = f"[{gpt_version}]{_DATANAME}-{self.shard_num}/{shard_idx+1:0{digits}d}"
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
            sleep(21600)

        dataset = concatenate_datasets(finish_shard_ls)
        for idx, data in enumerate(dataset):
            data["caption_ls"] = [data["ko_caption"]]
            yield (idx, data)


# simple EDA and data check
"""
from tqdm import tqdm
import tiktoken
from collections import Counter
from langdetect import detect, detect_langs


def num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18"):
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
"""
