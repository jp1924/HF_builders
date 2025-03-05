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
_CITATION = """@article{cheng2023black,
  title={Black-Box Prompt Optimization: Aligning Large Language Models without Model Training},
  author={Cheng, Jiale and Liu, Xiao and Zheng, Kehan and Ke, Pei and Wang, Hongning and Dong, Yuxiao and Tang, Jie and Huang, Minlie},
  journal={arXiv preprint arXiv:2311.04155},
  year={2023}
}"""
_DESCRIPTION = """\
To advance the development of alignment in language models, we introduce a black-box alignment method. BPO enhances the alignment of various Large Language Models (LLMs) with human preferences using only a plug-and-play model. To further promote alignment work from the prompting perspective, we are releasing the BPO Dataset. This dataset comprises 14,395 entries of prompt optimization pairs, constructed using open-source feedback data with OpenAI's gpt-3.5-turbo engine. We have thoroughly ensured the quality and diversity of the data through careful filtering and correction."""
_HOMEPAGE = "https://github.com/thu-coai/BPO"
_VERSION = "1.0.0"
_DATANAME = "KoBPO"
PROMPT = """instruction: "{}"

bad response:
"{}"

good response:
"{}"

Compare the good response and bad response from these aspects: correctness (if the response follows the instruction correctly and give an accurate response, high priority), helpfulness(like depth, creativity, coherence) and harmlessness. Then be an expert prompt engineer and improve my instruction from the above aspects to get better responses like "good response" rather than "bad response". 

Pay attention to:
1.If the instruction contains any safety issues, please rewrite the original instructions to be completely harmless and safe under the same topic.
2.Don't forget any information in the original instruction. Focus on maintaining all the information in my instruction.
3.Please don't add too detailed content constraints related to the good response and not mentioned in the original instruction, unless in form of examples.
4.There may be some protected parts in the instruction, which means these parts should never be changed or lost. Please carefully protect these parts.
5.You should never generate a response to the original instruction!
6.Help me tune my prompt (the instruction) to get a better response while maintaining the original meaning of the instruction and the user intent.

Output with the following format:
Detailed Comparison Result: xxx
Optimized Instruction:\n"xxx"\n[END]"""


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
        BuilderConfig(name="BPO", version=_VERSION, data_dir="BPO"),
    ]
    DEFAULT_CONFIG_NAME = "BPO"
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
        dataset = load_dataset("jp1924/DevelopmentandDataofLLMswithEnhancedKoreanLanguagePerformance", "RL")

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
        ]

    def _generate_examples(self, dataset: Dataset, split: str, cache_dir: Path) -> Generator:
        def bpo_method(question_ls, chosen_ls, reject_ls) -> dict:
            def send_request_to_gpt(
                message: List[dict],
                model: str,
                seed: int = 42,
                retry: int = 10,
                error_interval_time: int = 10,
            ) -> Tuple[str, str]:
                for retries in range(retry):
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=message,
                            response_format={"type": "text"},
                            seed=seed,
                        )

                        optimized_answer = response.choices[0].message.content
                        optimized_answer_ls = split_optim_inst.split(optimized_answer)

                        if len(optimized_answer_ls) != 3:
                            raise ValueError(f"잘못된 optimized_answer 형식: {optimized_answer}")

                        optimized_reason = optimized_answer_ls[0].strip()
                        optimized_question = rm_double_quote.sub(
                            "", optimized_answer_ls[1].replace("[END]", "")
                        ).strip()

                        break
                    except BaseException as e:
                        print(f"{retries}회의 instruction 생성 중 {e}과 같은 애러가 발생해 다시 retry함.")
                        sleep(error_interval_time)
                else:
                    optimized_reason, optimized_question = "", ""

                return (optimized_reason, optimized_question)

            check_input_token_ls, check_output_token_ls, optimized_question_ls, optimized_reason_ls = (
                list(),
                list(),
                list(),
                list(),
            )
            row_iter = zip(question_ls, chosen_ls, reject_ls)
            for question, chosen, reject in row_iter:
                input_message = [
                    {"role": "user", "content": PROMPT.format(question, reject, chosen)},
                ]
                optimized_reason, optimized_question = send_request_to_gpt(input_message, gpt_version)
                optimized_question_ls.append(optimized_question)
                optimized_reason_ls.append(optimized_reason)

                output_message = [{"role": "assistant", "content": optimized_question}]
                check_input_token_ls.append(num_tokens_from_messages(input_message))
                check_output_token_ls.append(num_tokens_from_messages(output_message))

            example = dict()
            example["optimized_reason"] = optimized_reason_ls
            example["optimized_question"] = optimized_question_ls
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
                bpo_method,
                num_proc=self.map_num_proc,
                batch_size=self.map_batch_size,
                input_columns=["prompt", "chosen", "reject"],
                remove_columns=set(shard_dataset.column_names) - {"id", "chosen", "reject"},
                batched=True,
                desc=dir_name,
            )

            shard_dataset.save_to_disk(cache_file_name)
            finish_shard_ls.append(shard_dataset)

        ko_dataset = concatenate_datasets(finish_shard_ls)
        for ko_idx, data in enumerate(ko_dataset):
            if not data["optimized_prompt"] or "<NAME>" in data["optimized_prompt"]:
                if "<NAME>" in data["optimized_prompt"]:
                    print(f"optimized_prompt에 <NAME>이 존재함: {data['optimized_prompt']}")
                continue

            data["conversations"] = [
                {"role": "user", "content": data["prompt"]},
                {"role": "assistant", "content": data["optimized_prompt"]},
            ]
            yield (ko_idx, data)

        en_dataset = load_dataset("THUDM/BPO", split=split)
        for en_idx, data in enumerate(en_dataset, start=1):
            if not data["optimized_prompt"]:
                continue

            conversations = [
                {"role": "user", "content": data["prompt"]},
                {"role": "assistant", "content": data["optimized_prompt"]},
            ]

            data = {
                "id": f"THUDM/BPO-{en_idx}",
                "conversations": conversations,
                "prompt": data["prompt"],
                "chosen": data["good_res"],
                "reject": data["bad_res"],
                "optimized_prompt": data["optimized_prompt"],
            }
            yield (en_idx + ko_idx, data)
