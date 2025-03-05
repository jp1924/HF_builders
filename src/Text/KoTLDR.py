import os
import re
from pathlib import Path
from time import sleep
from typing import Generator, List, Tuple

import tiktoken
import weave
from datasets import (
    BuilderConfig,
    Dataset,
    DatasetInfo,
    Features,
    GeneratorBasedBuilder,
    Split,
    SplitGenerator,
    Value,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from openai import OpenAI

from transformers import set_seed


client = OpenAI()
set_seed(42)

GPT_VERSION = "gpt-4o-mini-2024-07-18"
_LICENSE = """Apache-2.0 license"""
_CITATION = """@misc{Trl, 
    title={TRL-lib/tldr · datasets at hugging face},
    url={https://huggingface.co/datasets/trl-lib/tldr},
    journal={trl-lib/tldr · Datasets at Hugging Face},
    author={Trl}
} """
_DESCRIPTION = """The TL;DR dataset is a processed version of Reddit posts, specifically curated to train models using the TRL library for summarization tasks. It leverages the common practice on Reddit where users append "TL;DR" (Too Long; Didn't Read) summaries to lengthy posts, providing a rich source of paired text data for training summarization models."""
_HOMEPAGE = "https://huggingface.co/datasets/trl-lib/tldr"
_VERSION = "1.0.0"
_DATANAME = "KoTLDR"
SYSTEM = """You are an English Translator, specialized in translating sentence from English into korea. Your main goals are to ensure grammatically correct translations and deliver text that feels natural and human-oriented. 

Instructions:
1. Translate the provided sentence from English to the korea.
2. Ensure that the translation maintains the meaning and context of the original text.
3. Use appropriate grammar, syntax, and idiomatic expressions to make the translation sound natural.
4. Avoid literal translations unless necessary to preserve the meaning.
5. If there are cultural references or idioms, adapt them to be understandable and relevant in the korea.
6. Keep the formatting and structure of the original text intact unless specified otherwise.
7. Review the translation for any errors or awkward phrasing before finalizing."""

rm_double_quote = re.compile('["“”]')
split_optim_inst = re.compile(r"Optimized Instruction.*?\n.*?(.*).*?\n.*?\[END\]")


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


class KoBPO(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(name="GRPO", version=_VERSION, data_dir="GRPO"),
    ]
    DEFAULT_CONFIG_NAME = "GRPO"
    VERSION = _VERSION

    def _info(self):
        features = Features(
            {
                "id": Value("string"),
                "conversations": [{"role": Value("string"), "content": Value("string")}],
                "chosen": Value("string"),
                "reject": Value("string"),
                "prompt": Value("string"),
                "optimized_prompt": Value("string"),
            }
        )

        self.shard_num = int(os.getenv("SHARD_NUM", "10"))
        self.gpt_version = os.getenv("GPT_VERSION", GPT_VERSION)
        self.map_batch_size = int(os.getenv("MAP_BATCH_SIZE", "10"))
        self.map_num_proc = int(os.getenv("MAP_NUM_PROC", "5"))

        return DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[SplitGenerator]:  # type: ignore
        cache_dir = Path(dl_manager.download_config.cache_dir, _DATANAME)
        dataset = load_dataset("trl-lib/tldr")

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "dataset": dataset["train"],
                    "split": "train",
                    "cache_dir": cache_dir,
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "dataset": dataset["validation"],
                    "split": "validation",
                    "cache_dir": cache_dir,
                },
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "dataset": dataset["test"],
                    "split": "test",
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
