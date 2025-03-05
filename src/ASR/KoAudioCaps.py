# MIT License

# Copyright (c) 2019 AudioCaps authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import io
import json
import os
from pathlib import Path
from time import sleep
from typing import List

import tiktoken
from datasets import (
    Audio,
    BuilderConfig,
    Dataset,
    DatasetInfo,
    Features,
    GeneratorBasedBuilder,
    Split,
    SplitGenerator,
    Value,
    Version,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from openai import OpenAI
from scipy.io import wavfile


client = OpenAI()

_CITATION = """\
@inproceedings{audiocaps,
  title={AudioCaps: Generating Captions for Audios in The Wild},
  author={Kim, Chris Dongjoo and Kim, Byeongchang and Lee, Hyunmin and Kim, Gunhee},
  booktitle={NAACL-HLT},
  year={2019}
}
"""

_DESCRIPTION = """\
We explore audio captioning: generating natural language description for any kind of audio in the wild. We contribute AudioCaps, a large-scale dataset of about 46K audio clips to human-written text pairs collected via crowdsourcing on the  AudioSet dataset. The collected captions of AudioCaps are indeed faithful for audio inputs. We provide the source code of the models to explore what forms of audio representation and captioning models are effective for the audio captioning.
"""

GPT_VERSION = "gpt-4o-mini-2024-07-18"
_DATANAME = "KoAudioCaps"
# copied from: https://community.openai.com/t/i-need-your-help-with-prompt/860497/7
SYSTEM = """You are an English Translator, specialized in translating text from English into korea. Your main goals are to ensure grammatically correct translations and deliver text that feels natural and human-oriented.

Instructions:
1. Translate the provided sentence from English to the korea.
2. Ensure that the translation maintains the meaning and context of the original text.
3. Use appropriate grammar, syntax, and idiomatic expressions to make the translation sound natural.
4. Avoid literal translations unless necessary to preserve the meaning.
5. If there are cultural references or idioms, adapt them to be understandable and relevant in the korea.
6. Keep the formatting and structure of the original text intact unless specified otherwise.
7. Review the translation for any errors or awkward phrasing before finalizing."""


_HOMEPAGE = "https://github.com/cdjkim/audiocaps"

_LICENSE = "MIT License"
SAMPLE_RATE = 48000


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


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class KoAudioCaps(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="default",
            data_dir="default",
            version=Version("1.0.0"),
            description=_DESCRIPTION,
        ),
        BuilderConfig(
            name="chat",
            data_dir="chat",
            version=Version("1.0.0"),
            description=_DESCRIPTION,
        ),
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        if self.config.name == "default":
            features = Features(
                {
                    "id": Value("int32"),
                    "youtube_id": Value("string"),
                    "start_time": Value("int32"),
                    "audio": Audio(SAMPLE_RATE),
                    "length": Value("int32"),
                    "caption": Value("string"),
                    "en_caption": Value("string"),
                }
            )
        elif self.config.name == "chat":
            features = Features(
                {
                    "id": Value("int32"),
                    "audio": Audio(SAMPLE_RATE),
                    "conversations": [{"role": Value("string"), "content": Value("string")}],
                    "length": Value("int32"),
                }
            )

        self.shard_num = int(os.getenv("SHARD_NUM", "10"))
        self.gpt_version = os.getenv("GPT_VERSION", GPT_VERSION)
        self.map_batch_size = int(os.getenv("MAP_BATCH_SIZE", "200"))
        self.map_num_proc = int(os.getenv("MAP_NUM_PROC", "4"))
        self.duration = int(os.getenv("AUDIOCAPS_DURATION", "10"))

        return DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        cache_dir = Path(dl_manager.download_config.cache_dir, _DATANAME)
        datasets = load_dataset("jp1924/AudioCaps")

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "dataset": datasets["train"],
                    "split": "train",
                    "cache_dir": cache_dir,
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "dataset": datasets["validation"],
                    "split": "validation",
                    "cache_dir": cache_dir,
                },
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "dataset": datasets["test"],
                    "split": "test",
                    "cache_dir": cache_dir,
                },
            ),
        ]

    def _generate_examples(self, dataset: Dataset, split: str, cache_dir: Path):
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
            finish_en_ls, finish_ko_ls, finish_audio_ls, finish_length_ls = list(), list(), list(), list()
            for en_caption in example["caption"]:
                input_message = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"{en_caption} to 한국어"},
                ]
                ko_caption = send_request_to_gpt(input_message, gpt_version)
                # 단순 비용 계산용
                output_message = [{"role": "assistant", "content": ko_caption}]

                check_input_token_ls.append(num_tokens_from_messages(input_message))
                check_output_token_ls.append(num_tokens_from_messages(output_message))

                finish_en_ls.append(en_caption)
                finish_ko_ls.append(ko_caption)

            for audio, start_time in zip(example["audio"], example["start_time"]):
                _start_time = int(start_time * SAMPLE_RATE)
                _end_time = int((start_time + self.duration) * SAMPLE_RATE)

                part_audio = audio["array"][_start_time:_end_time]

                audio_bytes = io.BytesIO()
                wavfile.write(audio_bytes, SAMPLE_RATE, part_audio)
                finish_audio_ls.append(audio_bytes.getvalue())
                finish_length_ls.append(part_audio.shape[0] / SAMPLE_RATE)

            example["audio"] = finish_audio_ls
            example["length"] = finish_length_ls
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
            if self.config.name == "chat":
                conversations = [
                    {"role": "user", "content": [{"type": "audio"}]},
                    {"role": "user", "content": [{"type": "text", "text": data["caption"]}]},
                ]
                for chat in conversations:
                    chat["context"] = json.dumps(chat["content"], ensure_ascii=False)

                data["conversations"] = conversations
            data["id"] = data["audiocap_id"]
            output_data = data
            yield (idx, output_data)
