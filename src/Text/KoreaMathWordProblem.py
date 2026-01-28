import csv
import io
import json
import os
import random
import time
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Dict, Generator, List

from datasets import (
    BuilderConfig,
    Dataset,
    DatasetInfo,
    Features,
    GeneratorBasedBuilder,
    Split,
    SplitGenerator,
    Value,
    load_from_disk,
)
from datasets import logging as ds_logging
from openai import OpenAI
from tqdm import tqdm


_DATANAME = "KoreaMathWordProblem"
_DESCRIPTION = "한국어 수학 문제에 대한 데이터"
_HOMEPAGE = "https://github.com/jkc-ai/mwp-korean-data-2021"
_LICENSE = "https://github.com/jkc-ai/mwp-korean-data-2021/blob/main/LICENSE"

URLS = {
    "JENTI-Train": "https://raw.githubusercontent.com/jkc-ai/mwp-korean-data-2021/refs/heads/main/public_mwp_data.json",
    "TUNIB-Train": "https://raw.githubusercontent.com/tunib-ai/KMWP/refs/heads/main/data/train.csv",
}

model = "gpt-4o-mini-2024-07-18"
client = OpenAI()

ds_logging.set_verbosity_info()
logger = ds_logging.get_logger("datasets")


class KoreaMathWordProblem(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(name="GRPO", version="1.0.0", description=_DESCRIPTION),
        BuilderConfig(name="SFT", version="1.0.0", description=_DESCRIPTION),
    ]
    DEFAULT_CONFIG_NAME = "GRPO"

    def _info(self):
        if self.config.name == "GRPO":
            features = Features(
                {
                    "id": Value("int64"),
                    "prompt": Value("string"),
                    "answer": Value("string"),
                    "metadata": {
                        "source": Value("string"),
                        "type": Value("string"),
                        "equation": Value("string"),
                    },
                }
            )
        elif self.config.name == "SFT":
            features = Features(
                {
                    "id": Value("int64"),
                    "conversations": [{"role": Value("string"), "content": Value("string")}],
                    "prompt": Value("string"),
                    "answer": Value("string"),
                    "metadata": {
                        "source": Value("string"),
                        "type": Value("string"),
                        "equation": Value("string"),
                        "original_answer": Value("string"),
                        "prompt_tokens": Value("int32"),
                        "completion_tokens": Value("int32"),
                        "model": Value("string"),
                    },
                }
            )

        self.gpt_version = os.getenv("GPT_VERSION", model)
        self.batch_chunk_size = int(os.getenv("BATCH_CHUNK_SIZE", "400"))
        self.train_ratio = float(os.getenv("TRAIN_RATIO", "0.9"))

        return DatasetInfo(
            description=self.config.description,
            version=self.config.version,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager) -> List[SplitGenerator]:
        download_files = dl_manager.download(URLS)
        cache_dir = Path(dl_manager.download_config.cache_dir, _DATANAME)

        # Load all data sources
        train_data_jenti, valid_data_jenti = self._load_jenti_data(download_files["JENTI-Train"])
        train_data_tunib, valid_data_tunib = self._load_tunib_data(download_files["TUNIB-Train"])

        split_generator_ls = [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "data_list": train_data_jenti + train_data_tunib,
                    "split": "train",
                    "cache_dir": cache_dir,
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "data_list": valid_data_jenti + valid_data_tunib,
                    "split": "validation",
                    "cache_dir": cache_dir,
                },
            ),
        ]

        return split_generator_ls

    def _load_jenti_data(self, filepath: str) -> List[Dict]:
        """Load JETI format data from JSON"""
        with open(filepath, encoding="utf-8-sig") as f:
            data_dict = json.load(f)

        data_list = []
        for idx, item in data_dict.items():
            data_list.append(
                {
                    "prompt": item["question"],
                    "answer": item["answer"],
                    "source": "JENTI",
                    "type": item["class"],
                    "equation": item.get("equation", "none"),
                }
            )
        data_by_type = defaultdict(list)
        for data in data_list:
            data_by_type[data["type"]].append(data)
        for type_name in sorted(data_by_type.keys()):
            items = data_by_type[type_name]
            print(f"{type_name}: {len(items)}")

        # Stratified split: 각 유형별로 train_ratio만큼 train으로, 나머지는 validation으로
        train_data = []
        valid_data = []

        random.seed(42)
        for type_name in sorted(data_by_type.keys()):
            type_samples = data_by_type[type_name]
            random.shuffle(type_samples)

            # 각 유형별로 train_ratio만큼 분할
            split_idx = int(len(type_samples) * self.train_ratio)
            train_samples = type_samples[:split_idx]
            valid_samples = type_samples[split_idx:]

            train_data.extend(train_samples)
            valid_data.extend(valid_samples)
        return train_data, valid_data

    def _load_tunib_data(self, filepath: str) -> List[Dict]:
        """Load TUNIB format data from CSV"""
        data_list = []

        # Map class number to Korean class name
        class_map = {
            "1": "산술연산",
            "2": "순서정하기",
            "3": "조합하기",
            "4": "수찾기1",
            "5": "수찾기2",
            "6": "수찾기3",
            "7": "크기비교",
            "8": "도형산술",
        }

        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data_list.append(
                    {
                        "prompt": row["problem"],
                        "answer": row["answer"],
                        "source": "TUNIB",
                        "type": class_map.get(row["class"], row["class"]),
                        "equation": row.get("code", "none"),
                    }
                )
        data_by_type = defaultdict(list)
        for data in data_list:
            data_by_type[data["type"]].append(data)
        for type_name in sorted(data_by_type.keys()):
            items = data_by_type[type_name]
            print(f"{type_name}: {len(items)}")

        # Stratified split: 각 유형별로 train_ratio만큼 train으로, 나머지는 validation으로
        train_data = []
        valid_data = []

        random.seed(42)
        for type_name in sorted(data_by_type.keys()):
            type_samples = data_by_type[type_name]
            random.shuffle(type_samples)

            # 각 유형별로 train_ratio만큼 분할
            split_idx = int(len(type_samples) * self.train_ratio)
            train_samples = type_samples[:split_idx]
            valid_samples = type_samples[split_idx:]

            train_data.extend(train_samples)
            valid_data.extend(valid_samples)
        return train_data, valid_data

    def _generate_examples(self, **kwargs) -> Generator:
        if self.config.name == "GRPO":
            for idx, data in enumerate(self._grpo_generate_examples(**kwargs)):
                yield idx, data
        elif self.config.name == "SFT":
            for idx, data in enumerate(self._sft_generate_examples(**kwargs)):
                yield idx, data

    def _grpo_generate_examples(self, data_list: List[Dict], split: str, cache_dir: Path) -> Generator:
        for idx, data in enumerate(data_list):
            yield {
                "id": idx,
                "prompt": data["prompt"],
                "answer": data["answer"],
                "metadata": {
                    "source": data["source"],
                    "type": data["type"],
                    "equation": data["equation"],
                },
            }

    def _sft_generate_examples(self, data_list: List[Dict], split: str, cache_dir: Path) -> Generator:
        def process_batches(request_chunk_ls: List[List[str]], depth: int = 0, max_depth: int = 5) -> List[Dict]:
            final_results, retry_ls, batch_map = [], [], {}
            if depth == max_depth:
                print(f"⚠️ Max depth {max_depth} reached. Abandoning {len(request_chunk_ls)} batches.")
                return []

            # Create and submit batches
            for request_ls in tqdm(request_chunk_ls, desc=f"Creating batches at depth-{depth}"):
                request_file = io.BytesIO()
                request_file.write("\n".join(filter(None, request_ls)).encode("utf-8"))

                batch_file = client.files.create(file=request_file, purpose="batch")
                batch = client.batches.create(
                    input_file_id=batch_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                )
                batch_map[batch.id] = request_ls
                request_file.close()

            print(f"✓ Depth {depth}: Created {len(batch_map)} batches: {list(batch_map.keys())}")
            p_bar = tqdm(total=len(batch_map), desc=f"depth-{depth}")

            while batch_map:
                completed_batches = []
                for batch_id, request_ls in batch_map.items():
                    batch_status = client.batches.retrieve(batch_id).status

                    if batch_status == "completed":
                        output_file_id = client.batches.retrieve(batch_id).output_file_id
                        batch_save_data = []

                        if output_file_id:
                            batch_result = client.files.content(output_file_id).content
                            batch_ls = map(json.loads, batch_result.decode("utf-8").splitlines())

                            for result_data in batch_ls:
                                custom_id = str(result_data.get("custom_id"))

                                if result_data.get("error"):
                                    original_request = next(
                                        (req for req in request_ls if f'"custom_id": "{custom_id}"' in req), None
                                    )
                                    if original_request:
                                        retry_ls.append(original_request)
                                    continue

                                if custom_id not in original_data_map:
                                    print(f"Warning: ID {custom_id} not found in original map.")
                                    continue

                                original_row = original_data_map[custom_id]
                                body = result_data["response"]["body"]

                                formatted_data = {
                                    "id": int(custom_id),
                                    "prompt": original_row["prompt"],
                                    "answer": body["choices"][0]["message"]["content"],
                                    "conversations": [
                                        {"role": "user", "content": original_row["prompt"]},
                                        {"role": "assistant", "content": body["choices"][0]["message"]["content"]},
                                    ],
                                    "metadata": {
                                        "source": original_row["source"],
                                        "type": original_row["type"],
                                        "equation": original_row["equation"],
                                        "original_answer": original_row["answer"],
                                        "prompt_tokens": body["usage"]["prompt_tokens"],
                                        "completion_tokens": body["usage"]["completion_tokens"],
                                        "model": body["model"],
                                    },
                                }
                                batch_save_data.append(formatted_data)

                        if batch_save_data:
                            batch_cache_dir = cache_path.joinpath(batch_id)
                            Dataset.from_list(batch_save_data).save_to_disk(batch_cache_dir.as_posix())
                            final_results.extend(batch_save_data)

                        completed_batches.append(batch_id)
                        p_bar.update(1)

                    elif batch_status in ["expired", "failed", "cancelled"]:
                        print(f"❌ Batch {batch_id} {batch_status}")
                        if batch_status in ["expired", "failed"]:
                            retry_ls.extend(request_ls)
                        completed_batches.append(batch_id)
                        p_bar.update(1)

                for batch_id in completed_batches:
                    del batch_map[batch_id]

                if batch_map:
                    time.sleep(10)

            p_bar.close()

            if retry_ls:
                retry_chunk_ls = list(zip_longest(*[iter(retry_ls)] * self.batch_chunk_size, fillvalue=None))
                retry_chunk_ls = [list(filter(None, chunk)) for chunk in retry_chunk_ls]
                retry_results = process_batches(retry_chunk_ls, depth=depth + 1, max_depth=max_depth)
                final_results.extend(retry_results)

            return final_results

        cache_path = cache_dir.joinpath(split)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Load cached results
        processed_ids = set()
        for batch_dir in cache_path.iterdir():
            if batch_dir.is_dir():
                try:
                    cached_shard = load_from_disk(batch_dir.as_posix())
                    for row in cached_shard:
                        processed_ids.add(row["id"])
                        yield row
                except Exception as e:
                    print(f"⚠️ Failed to load cache from {batch_dir}: {e}")

        # Filter unprocessed data
        data_to_process = [{**data, "id": idx} for idx, data in enumerate(data_list) if idx not in processed_ids]

        if len(data_to_process) == 0:
            return

        # Prepare batch requests
        request_ls = []
        original_data_map = {str(row["id"]): row for row in data_to_process}

        for row in data_to_process:
            request_obj = {
                "custom_id": str(row["id"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.gpt_version,
                    "messages": [{"role": "user", "content": row["prompt"]}],
                },
            }
            request_ls.append(json.dumps(request_obj))

        # Chunk requests for batch processing
        request_chunk_ls = [
            list(filter(None, chunk))
            for chunk in zip_longest(*[iter(request_ls)] * self.batch_chunk_size, fillvalue=None)
        ]

        # Process batches and yield results
        new_results = process_batches(request_chunk_ls)
        for data in new_results:
            yield data
