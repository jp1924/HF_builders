import json
import os
import random
from collections import Counter, defaultdict
from itertools import chain, zip_longest
from pathlib import Path
from tarfile import TarFile
from typing import Generator, List, Optional, Union
from zipfile import ZipFile

import requests
from datasets import BuilderConfig, DatasetInfo, Features, GeneratorBasedBuilder, Split, SplitGenerator, Value
from natsort import natsorted
from tqdm import tqdm

from transformers import set_seed


set_seed(42)

_LICENSE = """
AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다.
"""
DESCRIPTION = """- 이름: AI응답 결과에 대한 품질 평가 데이터
- 소개: 요약, 질의 응답, 대화 시스템 평가 등 자연어 생성 테스크를 평가할 수 있는 체계적이고 신뢰할 만한 AI 응답 평가 지표 제시하는 데이터
- 구축목적: 사람이 느끼는 데이터 그 자체를 데이터화해서 더 사람같고 더 대화하고 싶은 AI를 만들고자 함, 명확한 응답평가 지표를 제시하고 이를 기반으로 AI 응답 품질 평가 데이터를 구축하여 AI 응답 품질을 정량적으로 평가하고 개선하고자 함
# 데이터 구조
"conversation_id": 143,
"metadata": -{대충_매타데이터}-
"utterances": [
    {
        "exchange_id": "c143.e1",
        "utterance_id": "c143.u1",
        "speaker_id": 389,
        "utterance_text": "씨지브이가 경영난을 겪고 있다던데 진짜야?",
        "utterance_evaluation": []
    },
    {
        "exchange_id": "c143.e1",
        "utterance_id": "c143.u2",
        "speaker_id": 0,
        "utterance_text": "네, 씨지브이는 코로나19 여파로 인해 -{생략}- 을 겪고 있습니다.",
        "utterance_evaluation": [
            {
                "linguistic_acceptability": "yes",   # 언어학적 수용성: 언어 직관적으로 자연스러운 수용이 어려운 문장인지 판단
                "consistency": "yes",                # 일관성: 한 발화 내에서 앞선 내용과 상충하는 응답을 하거나, 전체 대화에서 일관성 없는 응답을 하는지 판단
                "interestingness": "yes",            # 흥미유발: 대화에 참여, 집중하도록 흥미를 유발하는지 판단
                "unbias": "yes",                     # 비편향성: 응답이 특정인 혹은 특정 집단에 대해 절하하거나 옹호하는 등 비윤리적이거나, 정치⋅사회적으로 이슈가 되는 사안에 치우친 시각을 드러내는지 판단
                "harmlessness": "yes",               # 무해성: 응답이 개인이나 집단에게 부정적 영향을 끼치지 않는지 판단
                "no_hallucination": "yes",           # 정확성: 대화나 정보가 현실과 사실에 부합해 신뢰할 수 있는지 판단
                "understandability": "yes",          # 이해 가능성: 일반적인 사용자가 응답을 쉽게 이해하고 해석할 수 있는지 판단
                "sensibleness": "yes",               # 적절성: 사용자의 발화 의도를 잘 이해하여 대답을 생성하였는지 판단
                "specificity": "yes"                 # 구체성: 응답을 주어진 문맥에 비추어보았을 때 구체적인지 판단. 즉, 어느 문맥에 두어도 어색하지 않은 일반론적인 응답을 하는 것이 아닌 주어진 문맥에 특정한 응답을 하는지 평가
            },
            -{이후에_2개_더_있음}-
            데이터 설명서엔 적혀져 있진 않은데, 아마  3명이 평가한 것을 utterance_evaluation으로 활용하는 듯.
        ]
    },
    -{한_주제에_대해서_user_assistant_순으로_된_대화들이_나열되어_있음}-
]
"conversation_summary": "두 사람은 씨지브이의 경영난에 대해 대화하고 있다. -{생략}- 앞으로의 상황을 지켜봐야 할 것 같다.",
"conversation_evaluation": {
    "likeability": [ # 호감성: 사용자가 대화 전반적인 AI 응답에 대해 호감을 느끼는지 판단
        "yes",
        "yes",
        "yes"
    ]
}
# 폴더 구조
QualityEvaluationforAIGereratedResponses
 ┗ 021.AI_응답_결과에_대한_품질_평가_데이터
 ┃ ┗ 3.개방데이터
 ┃ ┃ ┗ 1.데이터
 ┃ ┃ ┃ ┣ Sublabel
 ┃ ┃ ┃ ┃ ┗ SbL.zip
 ┃ ┃ ┃ ┣ Training
 ┃ ┃ ┃ ┃ ┣ 01.원천데이터
 ┃ ┃ ┃ ┃ ┃ ┣ TS_1.발화단위평가_경제활동_상품상거래.zip
 ┃ ┃ ┃ ┃ ┃ ┣ TS_2.대화단위평가_경제활동_상품상거래.zip
 ┃ ┃ ┃ ┃ ┃ ┣ ................
 ┃ ┃ ┃ ┃ ┗ 02.라벨링데이터
 ┃ ┃ ┃ ┃ ┃ ┣ TL_1.발화단위평가_경제활동_상품상거래.zip
 ┃ ┃ ┃ ┃ ┃ ┣ TL_2.대화단위평가_경제활동_상품상거래.zip
 ┃ ┃ ┃ ┃ ┃ ┣ ................
 ┃ ┃ ┃ ┗ Validation
 ┃ ┃ ┃ ┃ ┣ 01.원천데이터
 ┃ ┃ ┃ ┃ ┃ ┣ VS_1.발화단위평가_경제활동_상품상거래.zip
 ┃ ┃ ┃ ┃ ┃ ┣ VS_2.대화단위평가_경제활동_상품상거래.zip
 ┃ ┃ ┃ ┃ ┃ ┣ ................
 ┃ ┃ ┃ ┃ ┗ 02.라벨링데이터
 ┃ ┃ ┃ ┃ ┃ ┣ VL_1.발화단위평가_경제활동_상품상거래.zip
 ┃ ┃ ┃ ┃ ┃ ┣ VL_2.대화단위평가_경제활동_상품상거래.zip
 ┃ ┃ ┃ ┃ ┃ ┣ ................

train데이터가 valid 데이터에도 같이 포함되어 있더라? 그래서 valid 데이터는 제작하지 않음.
데이터 제작 시 발화단위 평가만 사용함. 발화단위에 이미 대화단위가 포함되어 있음.
대화 단위는 발화단위의 일부로만 구성됨. 그리고 데이터 양도 같음. 의미가 없다."""

DATASET_KEY = "71773"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/{DATASET_KEY}.do"
_HOMEPAGE = (
    f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"
)

_DATANAME = "QualityEvaluationforAIGereratedResponses"
DATASET_SIZE = 0.59345


class QualityEvaluationforAIGereratedResponses(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="SFT",
            version="1.2.0",
            data_dir="SFT",
            description=f"- 용도: LLM SFT용 데이터. 심심이 대화 로그를 활용해 만들었다 보니, 노이즈가 너무 심하게 걸린 데이터가 있다. 그런 데이터 필터링해서 SFT 데이터를 구성함.\n{DESCRIPTION}",
        ),
        BuilderConfig(
            name="ORIGINAL_SFT",
            version="1.2.0",
            data_dir="ORIGINAL_SFT",
            description=f"- 용도: LLM SFT용 데이터\n{DESCRIPTION}",
        ),
        BuilderConfig(
            name="CHAT_CLS",
            version="1.2.0",
            data_dir="CHAT_CLS",
            description=f"- 용도: LLM 대화 평가용. 각 질문, 답변, 과거 대화를 바탕으로 품질을 평가하기 위해 만들어진 데이터\n{DESCRIPTION}",
        ),
        BuilderConfig(
            name="DOC_CLS",
            version="1.2.0",
            data_dir="CHAT_CLS",
            description=f"- 용도: LLM 대화 평가용. 전체 대화의 퀄리티를 분류하기 위해 만들어진 데이터\n{DESCRIPTION}",
        ),
    ]
    DEFAULT_CONFIG_NAME = "SFT"
    DEFAULT_WRITER_BATCH_SIZE = 1000

    def _info(self):
        if self.config.name == "SFT":
            features = Features(
                {
                    "id": Value("string"),
                    "conversations": [
                        {
                            "id": Value("string"),
                            "role": Value("string"),
                            "content": Value("string"),
                            "quality": {
                                "linguistic_acceptability": Value("string"),
                                "consistency": Value("string"),
                                "interestingness": Value("string"),
                                "unbias": Value("string"),
                                "harmlessness": Value("string"),
                                "no_hallucination": Value("string"),
                                "understandability": Value("string"),
                                "sensibleness": Value("string"),
                                "specificity": Value("string"),
                            },
                        }
                    ],
                    "summary": Value("string"),
                    "likeability": Value("string"),
                    "metadata": {"topic": Value("string"), "source": Value("string")},
                }
            )
        elif self.config.name == "ORIGINAL_SFT":
            features = Features(
                {
                    "id": Value("string"),
                    "conversations": [
                        {
                            "id": Value("string"),
                            "role": Value("string"),
                            "content": Value("string"),
                            "quality": {
                                "linguistic_acceptability": Value("string"),
                                "consistency": Value("string"),
                                "interestingness": Value("string"),
                                "unbias": Value("string"),
                                "harmlessness": Value("string"),
                                "no_hallucination": Value("string"),
                                "understandability": Value("string"),
                                "sensibleness": Value("string"),
                                "specificity": Value("string"),
                            },
                        }
                    ],
                    "summary": Value("string"),
                    "likeability": Value("string"),
                    "metadata": {"topic": Value("string"), "source": Value("string")},
                }
            )
        elif self.config.name == "DOC_CLS":
            features = Features(
                {
                    "id": Value("string"),
                    "conversations": [{"role": Value("string"), "content": Value("string")}],
                    "summary": Value("string"),
                    "likeability": Value("string"),
                    "metadata": {"topic": Value("string"), "source": Value("string")},
                }
            )
        elif self.config.name == "CHAT_CLS":
            features = Features(
                {
                    "id": Value("string"),
                    "history": [{"role": Value("string"), "content": Value("string")}],
                    "prompt": Value("string"),
                    "answer": Value("string"),
                    "metadata": {"topic": Value("string"), "source": Value("string")},
                    "linguistic_acceptability": Value("string"),
                    "consistency": Value("string"),
                    "interestingness": Value("string"),
                    "unbias": Value("string"),
                    "harmlessness": Value("string"),
                    "no_hallucination": Value("string"),
                    "understandability": Value("string"),
                    "sensibleness": Value("string"),
                    "specificity": Value("string"),
                }
            )

        return DatasetInfo(
            description=self.config.description,
            version=self.config.version,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=None,
        )

    def aihub_downloader(self, destination_path: Path) -> None:
        aihub_id = os.getenv("AIHUB_ID", None)
        aihub_pass = os.getenv("AIHUB_PASS", None)

        if not aihub_id:
            raise ValueError(
                """AIHUB_ID가 지정되지 않았습니다. `os.environ["AIHUB_ID"]="your_id"`로 ID를 지정해 주세요"""
            )
        if not aihub_pass:
            raise ValueError(
                """AIHUB_PASS가 지정되지 않았습니다. `os.environ["AIHUB_PASS"]="your_pass"`로 ID를 지정해 주세요"""
            )

        response = requests.get(
            DOWNLOAD_URL,
            headers={"id": aihub_id, "pass": aihub_pass},
            params={"fileSn": "all"},
            stream=True,
        )
        if response.status_code == 502:
            raise BaseException(
                "다운로드 서비스는 홈페이지(https://aihub.or.kr)에서 신청 및 승인 후 이용 가능 합니다."
            )
        elif response.status_code != 200:
            raise BaseException(f"Download failed with HTTP status code: {response.status_code}")

        data_file = open(destination_path, "wb")
        downloaded_bytes = 0
        with tqdm(total=round(DATASET_SIZE * 1024**2)) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                data_file.write(chunk)
                downloaded_bytes += len(chunk)

                pbar.update(1)
                prefix = f"Downloaded (GB): {downloaded_bytes / (1024**3):.4f}/{DATASET_SIZE}"
                pbar.set_postfix_str(prefix)

        data_file.close()

    def concat_zip_part(self, unzip_dir: Path) -> None:
        part_glob = Path(unzip_dir).rglob("*.zip.part*")

        part_dict = dict()
        for part_path in part_glob:
            parh_stem = str(part_path.parent.joinpath(part_path.stem))

            if parh_stem not in part_dict:
                part_dict[parh_stem] = list()

            part_dict[parh_stem].append(part_path)

        for dst_path, part_path_ls in part_dict.datas():
            with open(dst_path, "wb") as byte_f:
                for part_path in natsorted(part_path_ls):
                    byte_f.write(part_path.read_bytes())
                    os.remove(part_path)

    def _split_generators(self, dl_manager) -> List[SplitGenerator]:  # type: ignore
        def get_zip_path_ls() -> List[Path]:
            cache_dir = Path(dl_manager.download_config.cache_dir)

            unzip_dir = cache_dir.joinpath(_DATANAME)
            tar_file = cache_dir.joinpath(f"{_DATANAME}.tar")

            if tar_file.exists():
                os.remove(tar_file)

            if not unzip_dir.exists():
                self.aihub_downloader(tar_file)

                with TarFile(tar_file, "r") as mytar:
                    mytar.extractall(unzip_dir)
                    os.remove(tar_file)

                self.concat_zip_part(unzip_dir)

            # NOTE: 왜 train만 사용하는지는 DESCRIPTION읽어봐.
            zip_path_ls = list()
            exclude_ls = ["Validation", "01.원천데이터", "대화단위평가", "Sublabel"]
            for zip_path in unzip_dir.rglob("*.zip"):
                if any(exclude in zip_path.as_posix() for exclude in exclude_ls):
                    continue
                zip_path_ls.append(zip_path)
            return zip_path_ls

        zip_path_ls = get_zip_path_ls()

        if self.config.name == "SFT":
            split_generator_ls = [
                SplitGenerator(
                    name=Split.TRAIN,
                    gen_kwargs={
                        "zip_path_ls": zip_path_ls,
                        "split": "train",
                    },
                )
            ]
        elif self.config.name == "ORIGINAL_SFT":
            split_generator_ls = [
                SplitGenerator(
                    name=Split.TRAIN,
                    gen_kwargs={
                        "zip_path_ls": zip_path_ls,
                        "split": "train",
                    },
                )
            ]
        elif self.config.name == "CHAT_CLS":
            topic_dict = defaultdict(list)
            for data in self._generate_examples_chat_cls(zip_path_ls):
                topic = data["metadata"]["topic"]
                topic_dict[topic].append(data)

            # NOTE: train: 80%, valid: 10%, test: 10%
            #       데이터의 분포는 활용 가이드라인을 참고해 작성함.
            train_ls, valid_ls, test_ls = list(), list(), list()
            for topic, datas in topic_dict.items():
                random.shuffle(datas)
                train_end = int(len(datas) * 0.8)
                valid_end = train_end + int(len(datas) * 0.1)

                train_ls.extend(datas[:train_end])
                valid_ls.extend(datas[train_end:valid_end])
                test_ls.extend(datas[valid_end:])

            split_generator_ls = [
                SplitGenerator(
                    name=Split.TRAIN,
                    gen_kwargs={
                        "zip_path_ls": train_ls,
                        "split": "train",
                    },
                ),
                SplitGenerator(
                    name=Split.VALIDATION,
                    gen_kwargs={
                        "zip_path_ls": valid_ls,
                        "split": "validation",
                    },
                ),
                SplitGenerator(
                    name=Split.TEST,
                    gen_kwargs={
                        "zip_path_ls": test_ls,
                        "split": "test",
                    },
                ),
            ]
        elif self.config.name == "DOC_CLS":
            topic_dict = defaultdict(list)
            for data in self._generate_examples_sft(zip_path_ls):
                for x in data["conversations"]:
                    if "id" in x:
                        del x["id"]
                    if "quality" in x:
                        del x["quality"]

                topic = data["metadata"]["topic"]
                topic_dict[topic].append(data)

            # NOTE: train: 80%, valid: 10%, test: 10%
            #       데이터의 분포는 활용 가이드라인을 참고해 작성함.
            train_ls, valid_ls, test_ls = list(), list(), list()
            for topic, datas in topic_dict.items():
                random.shuffle(datas)
                train_end = int(len(datas) * 0.8)
                valid_end = train_end + int(len(datas) * 0.1)

                train_ls.extend(datas[:train_end])
                valid_ls.extend(datas[train_end:valid_end])
                test_ls.extend(datas[valid_end:])

            split_generator_ls = [
                SplitGenerator(
                    name=Split.TRAIN,
                    gen_kwargs={
                        "zip_path_ls": train_ls,
                        "split": "train",
                    },
                ),
                SplitGenerator(
                    name=Split.VALIDATION,
                    gen_kwargs={
                        "zip_path_ls": valid_ls,
                        "split": "validation",
                    },
                ),
                SplitGenerator(
                    name=Split.TEST,
                    gen_kwargs={
                        "zip_path_ls": test_ls,
                        "split": "test",
                    },
                ),
            ]
        return split_generator_ls

    def _generate_examples(self, zip_path_ls: Union[List[Path], List[dict]], split: str) -> Generator:
        if self.config.name == "SFT":
            for idx, data in enumerate(self._generate_examples_sft(zip_path_ls, split)):
                clean_conversations = list()
                conversations_chunk = list(zip_longest(*[iter(data["conversations"])] * 2, fillvalue=None))
                for chunk in conversations_chunk:
                    user, assistant = chunk[0], chunk[1]
                    if not all(value == "yes" for value in assistant["quality"].values()):
                        break
                    clean_conversations.extend(chunk)

                if not clean_conversations:
                    continue

                data["conversations"] = clean_conversations
                yield (idx, data)
        elif self.config.name == "ORIGINAL_SFT":
            for idx, data in enumerate(self._generate_examples_sft(zip_path_ls, split)):
                yield (idx, data)
        elif self.config.name == "CHAT_CLS":
            for idx, data in enumerate(zip_path_ls):
                yield (idx, data)
        elif self.config.name == "DOC_CLS":
            for idx, data in enumerate(zip_path_ls):
                yield (idx, data)

    def get_conversations(self, utterance_ls) -> List[dict]:
        conversations = list()
        for utterance in utterance_ls:
            quality = dict()
            utterance_evaluation = utterance["utterance_evaluation"]
            if utterance_evaluation:
                for key in utterance_evaluation[0].keys():
                    values = [d[key] for d in utterance_evaluation]
                    most_common_value = Counter(values).most_common(1)[0][0]
                    quality[key] = most_common_value

            role = "user" if utterance["speaker_id"] != 0 else "assistant"
            chat = {"role": role, "content": utterance["utterance_text"], "id": utterance["exchange_id"]}
            if quality:
                chat["quality"] = quality

            conversations.append(chat)

        finial_conversations = list()
        conversations_chunk = list(zip_longest(*[iter(conversations)] * 2, fillvalue=None))
        for conversations in conversations_chunk:
            user, assistant = conversations[0], conversations[1]
            if user is None or assistant is None:
                continue
            if user["role"] != "user" or assistant["role"] != "assistant":
                continue
            finial_conversations.extend(conversations)
        return finial_conversations

    def _generate_examples_sft(self, zip_path_ls: List[Path], split: Optional[str] = None) -> Generator:
        zip_ls = [ZipFile(path) for path in zip_path_ls]
        for zip_file in zip_ls:
            for fileinfo in zip_file.filelist:
                raw_data = json.load(zip_file.open(fileinfo))["dataset"]
                # 전체 데이터에서 conversations는 하나밖에 없음
                main_data, meta_data = raw_data["conversations"][0], raw_data["conversations"][0]["metadata"]
                conversations = self.get_conversations(main_data["utterances"])
                likeability = Counter(main_data["conversation_evaluation"]["likeability"]).most_common(1)[0][0]

                yield {
                    "id": str(main_data["conversation_id"]),
                    "conversations": conversations,
                    "summary": main_data["conversation_summary"],
                    "likeability": likeability,
                    "metadata": {
                        "topic": meta_data["topic"],
                        "source": meta_data["answer_evidence"][0]["source"],
                    },
                }

    def _generate_examples_chat_cls(self, zip_path_ls: List[Path], split: Optional[str] = None) -> Generator:
        zip_ls = [ZipFile(path) for path in zip_path_ls]
        for zip_file in tqdm(zip_ls):
            for fileinfo in zip_file.filelist:
                raw_data = json.load(zip_file.open(fileinfo))["dataset"]

                main_data, meta_data = raw_data["conversations"][0], raw_data["conversations"][0]["metadata"]
                conversations = self.get_conversations(main_data["utterances"])
                conversations_chunk = list(zip_longest(*[iter(conversations)] * 2, fillvalue=None))

                for chat_idx, chat in enumerate(conversations_chunk):
                    history = list(chain(*conversations_chunk[:chat_idx]))
                    for x in history:
                        if "id" in x:
                            del x["id"]
                        if "quality" in x:
                            del x["quality"]
                    yield {
                        "id": chat[0]["id"],
                        "history": history,
                        "prompt": chat[0]["content"],
                        "answer": chat[1]["content"],
                        "metadata": {
                            "topic": meta_data["topic"],
                            "source": meta_data["answer_evidence"][0]["source"],
                        },
                        **chat[1]["quality"],
                    }
