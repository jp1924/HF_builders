import random
from collections import defaultdict
from itertools import zip_longest
from typing import Generator, List

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
)
from natsort import natsorted


class LLMNationalGRPO(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(name="GRPO", version="1.0.0", description=""),
        BuilderConfig(name="SFT-N", version="1.0.0", description=""),
        BuilderConfig(name="SFT-P", version="1.0.0", description=""),
    ]
    DEFAULT_CONFIG_NAME = "GRPO"

    def _info(self) -> DatasetInfo:
        if self.config.name == "GRPO":
            features = {
                "id": Value("int32"),
                "prompt": Value("string"),
                "answer": Value("string"),
                "passage": Value("string"),
                "choice_ls": [Value("string")],
                "passage_ls": [Value("string")],
                "answerable": Value("bool"),
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
        elif self.config.name == "SFT-N":
            features = {
                "id": Value("string"),
                "conversations": [{"role": Value("string"), "content": Value("string")}],
                "prompt": Value("string"),
                "answer": Value("string"),
                "answer_passage": Value("string"),
                "passage": [Value("string")],
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
        elif self.config.name == "SFT-P":
            features = {
                "id": Value("string"),
                "conversations": [{"role": Value("string"), "content": Value("string")}],
                "prompt": Value("string"),
                "answer": Value("string"),
                "answer_passage": Value("string"),
                "passage": [Value("string")],
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

        return DatasetInfo(
            description=self.config.description,
            version=self.config.version,
            features=Features(features),
        )

    def _split_generators(self, _) -> List[SplitGenerator]:  # type: ignore
        datasets = load_dataset("jp1924/CorpusForLLMNationalRecordsAndArchives", "SFT")
        split_generator_ls = [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "dataset": datasets["train"],
                    "split": "train",
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "dataset": datasets["validation"],
                    "split": "validation",
                },
            ),
        ]

        return split_generator_ls

    def _generate_examples(self, **kwagrs) -> Generator:
        if self.config.name == "GRPO":
            for idx, data in enumerate(self._grpo_generate_examples(**kwagrs)):
                data["id"] = str(idx)
                yield idx, data
        elif self.config.name == "SFT-N":
            for idx, data in enumerate(self._sft_n_generate_examples(**kwagrs)):
                data["id"] = str(idx)
                yield idx, data
        elif self.config.name == "SFT-P":
            for idx, data in enumerate(self._sft_p_generate_examples(**kwagrs)):
                data["id"] = str(idx)
                yield idx, data

    def _grpo_generate_examples(self, dataset: Dataset, split: str):
        doc_idx_map = defaultdict(list)
        metadata_ls = dataset["metadata"]
        for idx, metadata in enumerate(metadata_ls):
            doc_idx_map[metadata["title"]].append(idx)

        filter_func = filter(lambda x: len(x[1]) >= 4, doc_idx_map.items())
        doc_ls = natsorted(filter_func, key=lambda x: len(x[1]), reverse=True)
        for idx, (doc_title, idx_ls) in enumerate(doc_ls):
            doc_ls[idx] = (
                doc_title,
                list(filter(lambda x: None not in x, zip_longest(*[iter(idx_ls)] * 4, fillvalue=None))),
            )

        data_count = 0
        negative_ratio = 0.2  # 비율: 답변 불가능 샘플을 생성할 확률
        negative_size = int(negative_ratio * len([y for _, x in doc_ls for y in x]))
        for doc_idx, (doc_title, chunk_ls) in enumerate(doc_ls):
            for group_idx, group_ls in enumerate(chunk_ls):
                select_data = dataset.select(group_ls)
                prompt_ls, answer_ls, passage_ls = select_data["prompt"], select_data["answer"], select_data["passage"]

                if data_count < negative_size:
                    if len(chunk_ls) > 1:
                        choice_group_ls = random.choice(chunk_ls[:group_idx] + chunk_ls[group_idx + 1 :])
                    else:
                        select_doc = random.choice(doc_ls[:doc_idx] + doc_ls[doc_idx + 1 :])[1]
                        choice_group_ls = random.choice(select_doc)
                    prompt = dataset[random.choice(choice_group_ls)]["prompt"]
                    answer, passage = None, None
                    answerable = 0
                else:
                    prompt = random.choice(prompt_ls)
                    select_idx = prompt_ls.index(prompt)
                    answer, passage = answer_ls[select_idx], passage_ls[select_idx]
                    answerable = 1  # 1은 답변 가능, 0은 답변 불가능

                yield {
                    "id": None,
                    "prompt": prompt,
                    "answer": answer,
                    "passage": passage,
                    "choice_ls": answer_ls,
                    "passage_ls": passage_ls,
                    "answerable": answerable,
                    "metadata": select_data["metadata"][0],
                }
                data_count += 1

    def _sft_n_generate_examples(self, dataset: Dataset, split: str):
        TOP_K = 3  # 정답 지문 1개 + 오답 지문 2개

        doc_idx_map = defaultdict(list)
        metadata_ls = dataset["metadata"]
        for idx, metadata in enumerate(metadata_ls):
            doc_idx_map[metadata["title"]].append(idx)

        # 문서 제목 리스트 (다른 문서에서 오답 지문을 가져올 때 사용)
        doc_titles = list(doc_idx_map.keys())

        # negative sampling: 전체 샘플 중 20%를 답변 불가로 만듦
        total_points = sum(len(idx_ls) for idx_ls in doc_idx_map.values())
        negative_ratio = 0.2
        negative_size = int(negative_ratio * total_points)
        negative_indices = set(random.sample(range(total_points), negative_size)) if negative_size > 0 else set()

        idx_counter = 0
        global_counter = 0
        for doc_title, idx_ls in doc_idx_map.items():
            for data_idx in idx_ls:
                # 정답 데이터
                correct_data = dataset[data_idx]
                correct_passage = correct_data["passage"]
                question = correct_data["prompt"]
                correct_answer = correct_data["answer"]

                is_negative = global_counter in negative_indices

                # 오답 지문 수집
                distractor_passages = []

                # 1. 같은 문서에서 오답 지문 수집
                same_doc_indices = [i for i in idx_ls if i != data_idx]
                if len(same_doc_indices) > 0:
                    # positive: 정답 제외 후 TOP_K-1 / negative: TOP_K
                    need = TOP_K if is_negative else TOP_K - 1
                    num_same_doc = min(need, len(same_doc_indices))
                    selected_same_doc = random.sample(same_doc_indices, num_same_doc)
                    distractor_passages.extend([dataset[i]["passage"] for i in selected_same_doc])

                # 2. 부족한 오답 지문을 다른 문서에서 수집
                remaining = (TOP_K if is_negative else TOP_K - 1) - len(distractor_passages)
                if remaining > 0:
                    other_docs = [title for title in doc_titles if title != doc_title]
                    for _ in range(remaining):
                        if len(other_docs) == 0:
                            break
                        random_doc = random.choice(other_docs)
                        random_idx = random.choice(doc_idx_map[random_doc])
                        candidate = dataset[random_idx]["passage"]
                        # 중복 피해주기
                        if candidate not in distractor_passages:
                            distractor_passages.append(candidate)

                # positive: 정답 + 오답 / negative: 오답들만
                if is_negative:
                    all_passages = distractor_passages[:TOP_K]  # TOP_K개의 오답 지문만 포함
                    answer_text = "죄송합니다. 현재 주어진 정보론 알맞은 답변을 할 수 없습니다."
                else:
                    all_passages = [correct_passage] + distractor_passages
                    random.shuffle(all_passages)
                    answer_text = correct_answer

                # 전체 지문을 하나의 문자열로 결합 (conversations에 사용)
                combined_passage = "\n".join(all_passages)

                yield {
                    "id": str(idx_counter),
                    "prompt": question,
                    "answer": answer_text,
                    "passage": all_passages,  # SFT config의 features에 맞게 리스트로 반환
                    "answer_passage": correct_passage if not is_negative else None,
                    "conversations": [
                        {"role": "system", "content": correct_data["conversations"][0]["content"]},
                        {"role": "user", "content": f"{combined_passage}\n\n{question}"},
                        {"role": "assistant", "content": answer_text},
                    ],
                    "metadata": correct_data["metadata"],
                }
                idx_counter += 1
                global_counter += 1

    def _sft_p_generate_examples(self, dataset: Dataset, split: str):
        TOP_K = 3  # 정답 지문 1개 + 오답 지문 2개

        doc_idx_map = defaultdict(list)
        metadata_ls = dataset["metadata"]
        for idx, metadata in enumerate(metadata_ls):
            doc_idx_map[metadata["title"]].append(idx)

        # 문서 제목 리스트 (다른 문서에서 오답 지문을 가져올 때 사용)
        doc_titles = list(doc_idx_map.keys())

        idx_counter = 0
        for doc_title, idx_ls in doc_idx_map.items():
            for data_idx in idx_ls:
                # 정답 데이터
                correct_data = dataset[data_idx]
                correct_passage = correct_data["passage"]
                question = correct_data["prompt"]
                correct_answer = correct_data["answer"]

                # 오답 지문 수집
                distractor_passages = []

                # 1. 같은 문서에서 오답 지문 수집
                same_doc_indices = [i for i in idx_ls if i != data_idx]
                if len(same_doc_indices) > 0:
                    # 같은 문서에서 랜덤하게 선택
                    num_same_doc = min(TOP_K - 1, len(same_doc_indices))
                    selected_same_doc = random.sample(same_doc_indices, num_same_doc)
                    distractor_passages.extend([dataset[i]["passage"] for i in selected_same_doc])

                # 2. 부족한 오답 지문을 다른 문서에서 수집
                remaining = (TOP_K - 1) - len(distractor_passages)
                if remaining > 0:
                    other_docs = [title for title in doc_titles if title != doc_title]
                    for _ in range(remaining):
                        if len(other_docs) == 0:
                            break
                        random_doc = random.choice(other_docs)
                        random_idx = random.choice(doc_idx_map[random_doc])
                        distractor_passages.append(dataset[random_idx]["passage"])

                # 모든 지문 합치기 (정답 + 오답)
                all_passages = [correct_passage] + distractor_passages
                random.shuffle(all_passages)

                # 전체 지문을 하나의 문자열로 결합
                combined_passage = "\n".join(all_passages)

                yield {
                    "id": str(idx_counter),
                    "prompt": question,
                    "answer": correct_answer,
                    "passage": all_passages,  # SFT config의 features에 맞게 리스트로 반환
                    "answer_passage": correct_passage,
                    "conversations": [
                        {"role": "system", "content": correct_data["conversations"][0]["content"]},
                        {"role": "user", "content": f"{combined_passage}\n\n{question}"},
                        {
                            "role": "assistant",
                            "content": correct_answer,
                        },
                    ],
                    "metadata": correct_data["metadata"],
                }
                idx_counter += 1
