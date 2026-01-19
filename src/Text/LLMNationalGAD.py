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
    load_dataset,
    load_from_disk,
)
from datasets import logging as ds_logging
from openai import OpenAI
from tqdm import tqdm


_DATANAME = "LLMNationalGAD"
model = "gpt-4o-mini-2024-07-18"

client = OpenAI()

ds_logging.set_verbosity_info()
logger = ds_logging.get_logger("datasets")


class LLMNationalGAD(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(name="GAD", version="1.0.0", description=""),
    ]
    DEFAULT_CONFIG_NAME = "GAD"

    def _info(self) -> DatasetInfo:
        if self.config.name == "GAD":
            features = {
                "id": Value("int32"),
                "prompt": Value("string"),
                "answer": Value("string"),
                "passage": Value("string"),
                "passage_ls": [Value("string")],
                "metadata": {
                    "publisher": Value("string"),
                    "date": Value("string"),
                    "collection_name": Value("string"),
                    "category_main": Value("string"),
                    "category_middle": Value("string"),
                    "registration_no": Value("string"),
                    "source_id": Value("string"),
                    "title": Value("string"),
                },
            }

        self.gpt_version = os.getenv("GPT_VERSION", model)
        self.batch_chunk_size = int(os.getenv("BATCH_CHUNK_SIZE", "400"))

        return DatasetInfo(
            description=self.config.description,
            version=self.config.version,
            features=Features(features),
        )

    def _split_generators(self, dl_manager) -> List[SplitGenerator]:  # type: ignore
        cache_dir = Path(dl_manager.download_config.cache_dir, _DATANAME)
        datasets = load_dataset("jp1924/CorpusForLLMNationalRecordsAndArchives", "SFT")

        for key in datasets.keys():
            dataset = datasets[key]
            dataset = dataset.remove_columns(["id"])
            dataset = dataset.add_column("id", list(range(len(dataset))))

            datasets[key] = dataset

        split_generator_ls = [
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
        ]

        return split_generator_ls

    def _generate_examples(self, **kwagrs) -> Generator:
        if self.config.name == "GAD":
            for idx, data in enumerate(self._raft_gad_generate_examples(**kwagrs)):
                data["id"] = str(idx)
                yield idx, data

    def _gad_generate_examples(self, dataset: Dataset, split: str, cache_dir: Path) -> Generator:
        def process_batches(request_chunk_ls: List[List[str]], depth: int = 0, max_depth: int = 5) -> List[Dict]:
            final_results, retry_ls, batch_map = [], [], {}
            if depth == max_depth:  # 최대 재시도 깊이 도달 시 중단
                print(f"⚠️ Max depth {max_depth} reached. Abandoning {len(request_chunk_ls)} batches.")
                return []

            # Batch 생성 및 전송
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
                        batch_save_data = []  # 현재 배치 저장을 위한 리스트

                        if output_file_id:
                            batch_result = client.files.content(output_file_id).content
                            batch_ls = map(json.loads, batch_result.decode("utf-8").splitlines())
                            # 결과 파싱 및 데이터 구성
                            for result_data in batch_ls:
                                custom_id = int(result_data.get("custom_id"))

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
                                    "id": original_row["id"],
                                    "prompt": original_row["prompt"],
                                    "answer": body["choices"][0]["message"]["content"],
                                    "passage": original_row["passage"],
                                    "metadata": {
                                        **original_row["metadata"],
                                        "original_answer": original_row["answer"],
                                        "prompt_tokens": body["usage"]["prompt_tokens"],
                                        "completion_tokens": body["usage"]["completion_tokens"],
                                    },
                                }
                                batch_save_data.append(formatted_data)

                        # 배치 단위 저장 (Cache Save)
                        if batch_save_data:
                            batch_cache_dir = cache_path.joinpath(batch_id)
                            Dataset.from_list(batch_save_data).save_to_disk(batch_cache_dir.as_posix())
                            final_results.extend(batch_save_data)  # 결과 반환용 리스트에 추가

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

        # 1. 기존 캐시 로드 및 처리된 ID 확인 (Resume Logic)
        processed_ids = set()
        # cache_path 내의 모든 디렉토리(batch_id)를 순회
        for batch_dir in cache_path.iterdir():
            if batch_dir.is_dir():
                try:
                    # 저장된 데이터셋 로드
                    cached_shard = load_from_disk(batch_dir.as_posix())
                    for row in cached_shard:
                        processed_ids.add(row["id"])
                        yield row
                except Exception as e:
                    print(f"⚠️ Failed to load cache from {batch_dir}: {e}")

        # 2. 처리되지 않은 데이터 필터링
        # 전체 데이터셋에서 이미 처리된 ID를 제외
        dataset_to_process = dataset.filter(lambda x: x["id"] not in processed_ids)

        if len(dataset_to_process) == 0:
            print("✓ All data already processed and cached.")
            return

        request_ls = []
        original_data_map = {int(row["id"]): row for row in dataset_to_process}

        for row in dataset_to_process:
            content = f"{row['passage']}\n\n{row['prompt']}"
            request_obj = {
                "custom_id": str(row["id"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.gpt_version,
                    "messages": [{"role": "user", "content": content}],
                },
            }
            request_ls.append(json.dumps(request_obj))

        # 5. Batch Chunking (API 제한 고려, 예: 50,000개)
        request_chunk_ls = [
            list(filter(None, chunk))
            for chunk in zip_longest(*[iter(request_ls)] * self.batch_chunk_size, fillvalue=None)
        ]

        # 6. 배치 처리 실행 및 결과 Yield
        # process_batches 내부에서 이미 저장은 완료되었으므로, 반환된 리스트를 yield만 하면 됨
        new_results = process_batches(request_chunk_ls)

        for data in new_results:
            yield data

    def _raft_gad_generate_examples(self, dataset: Dataset, split: str, cache_dir: Path) -> Generator:
        # 기존 코드 B의 로직 이어서
        def process_batches(request_chunk_ls: List[List[str]], depth: int = 0, max_depth: int = 5) -> List[Dict]:
            final_results, retry_ls, batch_map = [], [], {}
            if depth == max_depth:
                print(f"⚠️ Max depth {max_depth} reached. Abandoning {len(request_chunk_ls)} batches.")
                return []

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
                                    "id": original_row["id"],
                                    "prompt": original_row["prompt"],
                                    "answer": body["choices"][0]["message"]["content"],
                                    "passage": original_row["passage"],  # 리스트로 저장 (오답 지문 포함)
                                    "passage_ls": original_row["passage_ls"],
                                    "metadata": {
                                        **original_row["metadata"],
                                        "original_answer": original_row["answer"],
                                        "prompt_tokens": body["usage"]["prompt_tokens"],
                                        "completion_tokens": body["usage"]["completion_tokens"],
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

        # RAFT: 오답 지문 추가 로직 (코드 A의 로직 통합)
        TOP_K = 3  # 정답 지문 1개 + 오답 지문 2개

        doc_idx_map = defaultdict(list)
        metadata_ls = dataset["metadata"]
        for idx, metadata in enumerate(metadata_ls):
            doc_idx_map[metadata["title"]].append(idx)

        doc_titles = list(doc_idx_map.keys())

        raft_data = []
        idx_counter = 0
        for doc_title, idx_ls in doc_idx_map.items():
            for data_idx in idx_ls:
                correct_data = dataset[data_idx]
                correct_passage = correct_data["passage"]
                question = correct_data["prompt"]
                correct_answer = correct_data["answer"]

                distractor_passages = []

                same_doc_indices = [i for i in idx_ls if i != data_idx]
                if len(same_doc_indices) > 0:
                    num_same_doc = min(TOP_K - 1, len(same_doc_indices))
                    selected_same_doc = random.sample(same_doc_indices, num_same_doc)
                    distractor_passages.extend([dataset[i]["passage"] for i in selected_same_doc])

                remaining = (TOP_K - 1) - len(distractor_passages)
                if remaining > 0:
                    other_docs = [title for title in doc_titles if title != doc_title]
                    for _ in range(remaining):
                        if len(other_docs) == 0:
                            break
                        random_doc = random.choice(other_docs)
                        random_idx = random.choice(doc_idx_map[random_doc])
                        distractor_passages.append(dataset[random_idx]["passage"])

                all_passages = [correct_passage] + distractor_passages
                random.shuffle(all_passages)

                raft_data.append(
                    {
                        "id": str(idx_counter),
                        "prompt": question,
                        "answer": correct_answer,
                        "passage": correct_passage,
                        "passage_ls": all_passages,
                        "metadata": correct_data["metadata"],
                    }
                )
                idx_counter += 1

        dataset = Dataset.from_list(raft_data)

        cache_path = cache_dir.joinpath(split)
        cache_path.mkdir(parents=True, exist_ok=True)

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

        dataset_to_process = dataset.filter(lambda x: x["id"] not in processed_ids)

        if len(dataset_to_process) == 0:
            print("✓ All data already processed and cached.")
            return

        request_ls = []
        original_data_map = {str(row["id"]): row for row in dataset_to_process}

        for row in dataset_to_process:
            content = f"{row['passage']}\n\n{row['prompt']}"  # passage가 리스트이므로 결합
            request_obj = {
                "custom_id": str(row["id"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.gpt_version,
                    "messages": [{"role": "user", "content": content}],
                },
            }
            request_ls.append(json.dumps(request_obj))

        request_chunk_ls = [
            list(filter(None, chunk))
            for chunk in zip_longest(*[iter(request_ls)] * self.batch_chunk_size, fillvalue=None)
        ]

        new_results = process_batches(request_chunk_ls)
        for data in new_results:
            yield data
