import json
from concurrent.futures import ThreadPoolExecutor, as_completed
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


# vLLM API 클라이언트 설정
client = OpenAI(
    base_url="http://0.0.0.0:8000/v1",
    api_key="",  # vLLM은 API 키가 필요 없지만 형식상 필요
)

_DATANAME = "KoAgentInstruct"
MODEL_NAME = "yanolja/YanoljaNEXT-Rosetta-27B-2511"
MAX_WORKERS = 40  # 병렬 처리 워커 수


# Task별 context 정의
TASK_CONTEXTS = {
    # ...existing code...
    "rc": {
        "context": "Reading comprehension tasks involving processing and understanding text passages",
        "tone": "Analytical and precise",
        "glossary": {
            "assistant": "어시스턴트 답변",
            "user": "사용자 질문",
            "system": "시스템 지침",
        },
    },
    "open_domain_qa": {
        "context": "Open-domain question answering across wide range of topics",
        "tone": "Informative and comprehensive",
        "glossary": {
            "assistant": "어시스턴트 답변",
            "user": "사용자 질문",
            "system": "시스템 지침",
        },
    },
    "text_modification": {
        "context": "Text editing and modification to improve quality, tone, or fit specific context",
        "tone": "Clear and constructive",
        "glossary": {
            "assistant": "어시스턴트 답변",
            "user": "사용자 질문",
            "system": "시스템 지침",
        },
    },
    "follow_up": {  # Web Agent
        "context": "Web agent tasks performing autonomous actions like clicking and scrolling",
        "tone": "Instructional and action-oriented",
        "glossary": {
            "assistant": "어시스턴트 답변",
            "user": "사용자 질문",
            "system": "시스템 지침",
        },
    },
    "brain_teaser": {
        "context": "Brain teasers and puzzles requiring logical thinking and problem-solving",
        "tone": "Engaging and thought-provoking",
        "glossary": {
            "assistant": "어시스턴트 답변",
            "user": "사용자 질문",
            "system": "시스템 지침",
        },
    },
    "analytical_reasoning": {
        "context": "Analytical reasoning tasks identifying patterns and drawing logical conclusions",
        "tone": "Logical and systematic",
        "glossary": {
            "assistant": "어시스턴트 답변",
            "user": "사용자 질문",
            "system": "시스템 지침",
        },
    },
    "mcq": {
        "context": "Multiple choice questions for assessment and evaluation",
        "tone": "Clear and educational",
        "glossary": {
            "assistant": "어시스턴트 답변",
            "user": "사용자 질문",
            "system": "시스템 지침",
        },
    },
    "struct2text_flow": {  # Data To Text
        "context": "Converting structured data into human-readable textual summaries",
        "tone": "Clear and descriptive",
        "glossary": {
            "assistant": "어시스턴트 답변",
            "user": "사용자 질문",
            "system": "시스템 지침",
        },
    },
    "fermi": {
        "context": "Fermi estimation problems requiring rough estimates with justified assumptions",
        "tone": "Analytical and approximative",
        "glossary": {
            "assistant": "어시스턴트 답변",
            "user": "사용자 질문",
            "system": "시스템 지침",
        },
    },
    "code": {
        "context": "Programming tasks including writing, understanding, and debugging code",
        "tone": "Technical and precise",
        "glossary": {
            "assistant": "어시스턴트 답변",
            "user": "사용자 질문",
            "system": "시스템 지침",
        },
    },
    "text_extraction": {
        "context": "Extracting relevant information from larger text documents",
        "tone": "Precise and focused",
        "glossary": {
            "assistant": "어시스턴트 답변",
            "user": "사용자 질문",
            "system": "시스템 지침",
        },
    },
    "text_classification": {
        "context": "Classifying text documents into predefined categories",
        "tone": "Analytical and categorical",
        "glossary": {
            "assistant": "어시스턴트 답변",
            "user": "사용자 질문",
            "system": "시스템 지침",
        },
    },
    "rag": {
        "context": "Retrieval-augmented generation combining retrieval and generative models",
        "tone": "Informative and evidence-based",
        "glossary": {
            "assistant": "어시스턴트 답변",
            "user": "사용자 질문",
            "system": "시스템 지침",
        },
    },
    "fs_cot_flow": {  # Few Shot Reasoning (Chain-of-Thought)
        "context": "Few-shot reasoning learning new concepts from minimal examples",
        "tone": "Step-by-step and educational",
        "glossary": {
            "assistant": "어시스턴트 답변",
            "user": "사용자 질문",
            "system": "시스템 지침",
        },
    },
    "creative_content": {
        "context": "Creative content generation with novelty, value, and originality",
        "tone": "Creative, engaging, and imaginative",
        "glossary": {
            "assistant": "어시스턴트 답변",
            "user": "사용자 질문",
            "system": "시스템 지침",
        },
    },
}


def create_translation_system_prompt(task_name: str, target_language="Korean"):
    """Task별 야놀자 스타일의 번역 시스템 프롬프트 생성"""
    # ...existing code...
    context = TASK_CONTEXTS[task_name]

    system = [f"Translate the user's text to {target_language}."]
    for key, value in context.items():
        key_pascal = key.capitalize()
        if isinstance(value, dict):
            system.append(f"{key_pascal}:")
            for f, t in value.items():
                system.append(f"- {f} -> {t}")
        else:
            system.append(f"{key_pascal}: {value}")

    system.append("Output format: JSON")
    system.append("Provide the final translation immediately without any other text.")

    return "\n".join(system)


ds_logging.set_verbosity_info()
logger = ds_logging.get_logger("datasets")


class KoAgentInstruct(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(name="creative_content", version="1.0.0", description=""),
        BuilderConfig(name="text_modification", version="1.0.0", description=""),
        BuilderConfig(name="struct2text_flow", version="1.0.0", description=""),
        BuilderConfig(name="rc", version="1.0.0", description=""),
        BuilderConfig(name="rag", version="1.0.0", description=""),
        BuilderConfig(name="text_extraction", version="1.0.0", description=""),
        BuilderConfig(name="mcq", version="1.0.0", description=""),
        BuilderConfig(name="follow_up", version="1.0.0", description=""),
        BuilderConfig(name="analytical_reasoning", version="1.0.0", description=""),
        BuilderConfig(name="fermi", version="1.0.0", description=""),
        BuilderConfig(name="fs_cot_flow", version="1.0.0", description=""),
        BuilderConfig(name="code", version="1.0.0", description=""),
        BuilderConfig(name="brain_teaser", version="1.0.0", description=""),
        BuilderConfig(name="text_classification", version="1.0.0", description=""),
        BuilderConfig(name="open_domain_qa", version="1.0.0", description=""),
    ]
    DEFAULT_CONFIG_NAME = "open_domain_qa"

    def _info(self) -> DatasetInfo:
        features = {
            "id": Value("string"),
            "conversations": [{"role": Value("string"), "content": Value("string")}],
            "system": Value("string"),
            "prompt": Value("string"),
            "answer": Value("string"),
            "original": {
                "prompt": Value("string"),
                "answer": Value("string"),
                "system": Value("string"),
            },
        }

        return DatasetInfo(
            description=self.config.description,
            version=self.config.version,
            features=Features(features),
        )

    def _split_generators(self, dl_manager) -> List[SplitGenerator]:
        """원본 영어 데이터셋 로드 및 split 정보만 생성"""
        cache_dir = Path(dl_manager.download_config.cache_dir, _DATANAME)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 현재 config에 해당하는 영어 데이터셋 로드
        config_name = self.config.name
        logger.info(f"Loading English dataset for config: {config_name}")

        en_dataset = load_dataset(
            "microsoft/orca-agentinstruct-1M-v1",
            name="default",
            split=config_name if config_name != "code" else "code_",
        )

        # train/test split
        dataset_dict = en_dataset.train_test_split(test_size=100, shuffle=True, seed=42)

        split_generator_ls = [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "dataset": dataset_dict["train"],
                    "cache_dir": cache_dir,
                    "config_name": config_name,
                },
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "dataset": dataset_dict["test"],
                    "cache_dir": cache_dir,
                    "config_name": config_name,
                },
            ),
        ]

        return split_generator_ls

    def _translate_single_item(self, idx: int, data: Dict, config_name: str) -> Dict:
        """단일 아이템 번역 (캐싱 포함)"""
        item_id = f"{config_name}-{idx}"

        try:
            # Task별 시스템 프롬프트 생성
            trans_system = create_translation_system_prompt(config_name)

            conversations = json.loads(data["messages"])
            system_msg = conversations[0]["content"]
            user_msg = conversations[1]["content"]
            assistant_msg = conversations[2]["content"]

            has_system = bool(system_msg)

            # 번역할 소스 데이터 구성
            source = {}
            if has_system:
                source["system"] = system_msg
            source["user"] = user_msg
            source["assistant"] = assistant_msg

            # API 요청
            messages = [
                {"role": "system", "content": trans_system},
                {"role": "user", "content": json.dumps(source, ensure_ascii=False)},
            ]

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
            )

            # 응답 파싱
            translated_text = response.choices[0].message.content
            translated_data = json.loads(translated_text)

            # 결과 포맷팅
            if has_system:
                system_translated = translated_data.get("system", "")
                user_translated = translated_data.get("user", "")
                assistant_translated = translated_data.get("assistant", "")
                conversations_translated = [
                    {"role": "system", "content": system_translated},
                    {"role": "user", "content": user_translated},
                    {"role": "assistant", "content": assistant_translated},
                ]
            else:
                system_translated = ""
                user_translated = translated_data.get("user", "")
                assistant_translated = translated_data.get("assistant", "")
                conversations_translated = [
                    {"role": "user", "content": user_translated},
                    {"role": "assistant", "content": assistant_translated},
                ]

            return {
                "id": item_id,
                "conversations": conversations_translated,
                "system": system_translated,
                "prompt": user_translated,
                "answer": assistant_translated,
                "original": {
                    "system": system_msg,
                    "prompt": user_msg,
                    "answer": assistant_msg,
                },
                "success": True,
            }

        except Exception as e:
            logger.error(f"Error translating {item_id}: {e}")
            return {
                "id": item_id,
                "success": False,
                "error": str(e),
            }

    def _generate_examples(self, dataset: Dataset, cache_dir: Path, config_name: str) -> Generator:
        """각 example을 순회하며 번역 수행"""
        # 캐시 디렉토리 설정
        cache_path = cache_dir / "translated_by_config" / config_name
        cache_path.mkdir(parents=True, exist_ok=True)

        # 기존 캐시 로드
        cached_data = {}
        if cache_path.exists():
            try:
                cached_dataset = load_from_disk(cache_path.as_posix())
                for row in cached_dataset:
                    cached_data[row["id"]] = row
                logger.info(f"Loaded {len(cached_data)} cached items for {config_name}")
            except Exception as e:
                logger.info(f"No valid cache found: {e}")

        # 번역이 필요한 항목들 수집
        items_to_translate = []
        for idx, data in enumerate(dataset):
            item_id = f"{config_name}-{idx}"
            if item_id not in cached_data:
                items_to_translate.append((idx, data))

        # 병렬 번역 수행
        if items_to_translate:
            logger.info(f"Translating {len(items_to_translate)} new items for {config_name}...")

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(self._translate_single_item, idx, data, config_name): (idx, data)
                    for idx, data in items_to_translate
                }

                newly_translated = []
                failed_count = 0
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Translating {config_name}",
                ):
                    result = future.result()
                    if result.get("success", False):
                        cached_data[result["id"]] = result
                        newly_translated.append(result)
                    else:
                        failed_count += 1
                        logger.warning(f"Skipping failed translation: {result['id']}")

            logger.info(f"Successfully translated: {len(newly_translated)}, Failed: {failed_count}")

            # 새로 번역된 데이터 저장
            if newly_translated:
                all_cached = list(cached_data.values())
                Dataset.from_list(all_cached).save_to_disk(cache_path.as_posix())
                logger.info(f"Saved {len(all_cached)} total items for {config_name}")

        # 성공한 항목만 순서대로 yield
        output_idx = 0
        for idx in range(len(dataset)):
            item_id = f"{config_name}-{idx}"
            if item_id in cached_data:
                yield output_idx, cached_data[item_id]
                output_idx += 1
            else:
                logger.debug(f"Skipping {item_id} - not in cache (translation failed)")
