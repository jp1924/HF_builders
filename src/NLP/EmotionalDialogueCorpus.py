import json
import os
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from tarfile import TarFile
from typing import Generator, List
from zipfile import ZipFile

import requests
from datasets import BuilderConfig, DatasetInfo, Features, GeneratorBasedBuilder, Split, SplitGenerator, Value, Version
from kss import Kss
from natsort import natsorted
from tqdm import tqdm


_LICENSE = """
AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다.
"""
DESCRIPTION = """- 이름: 감성 대화 말뭉치
- 소개: 크라우드 소싱 수행으로 일반인 1,500명을 대상으로 하여 음성 15,700문장 및 코퍼스 27만 문장 구축 및 세대별 감성 대화 텍스트 구축을 통해 감성 대화 엔진을 개발하여 세대별 감성 대화 서비스 제공
- 구축목적: 감정 인식을 위한 데이터는 크롤링이 불가능하기 때문에 직접 제작해야 하는 희소성 있는 데이터임. 60가지의 세부 감정에 대한 자연어 처리 말뭉치를 확보함으로써 다양한 AI 산업에 활용이 가능함

# 데이터 구조
## 감성 대화 데이터
[
    {
        "profile": {
            "persona-id": "Pro_03802",
            "persona": {"persona-id": "A02_G01_C01", "human": ["A02", "G01"], "computer": ["C01"]},
            "emotion": {"emotion-id": "S06_D02_E31", "type": "E31", "situation": ["S06", "D02"]},
        },
        "talk": {
            "id": {"profile-id": "Pro_03802", "talk-id": "Pro_03802_00012"},
            "content": {
                "HS01": "회사에서 중요한 프로젝트를 혼자 하게 됐는데 솔직히 두렵고 무서워.",
                "SS01": "큰 프로젝트를 혼자 하셔서 고민이 많겠네요.",
                "HS02": "나에게 너무 크게 느껴지는 중요한 프로젝트라 버거운 느낌이 들어.",
                "SS02": "프로젝트를 잘하시기 위해서 어떤 걸 할 수 있나요?",
                "HS03": "동료 직원에게 도움을 요청해서 같이 해결해야겠어.",
                "SS03": "동료 직원에게 도움을 요청하기로 하셨군요.",
            },
        },
    },
    - {생략} -
]
# 폴더 구조
EmotionalDialogueCorpus
 ┗ 018.감성대화
 ┃ ┣ Training_221115_add
 ┃ ┃ ┣ 라벨링데이터
 ┃ ┃ ┃ ┗ 감성대화말뭉치(최종데이터)_Training.zip
 ┃ ┃ ┃   ┗ one_json_file
 ┃ ┃ ┗ 원천데이터
 ┃ ┃ ┃ ┗ 감성대화말뭉치(최종데이터)_Training.zip
 ┃ ┗ Validation_221115_add
 ┃ ┃ ┣ 라벨링데이터
 ┃ ┃ ┃ ┗ 감성대화말뭉치(최종데이터)_Validation.zip
 ┃ ┃ ┃   ┗ one_json_file
 ┃ ┃ ┗ 원천데이터
 ┃ ┃ ┃ ┗ 감성대화말뭉치(최종데이터)_Validation.zip
와 같이 구성되어 있다. 그릐고 유해질의 데이터는 데이터 목적하고 맞지 않아서 따로 뺌.
"""

DATASET_KEY = "86"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/{DATASET_KEY}.do"
_HOMEPAGE = (
    f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"
)

_DATANAME = "EmotionalDialogueCorpus"
DATASET_SIZE = 0.02035

# copied from: https://github.com/parkjg20/LLM-Depressed/tree/main/HuggingFaceTrain
SITUATION_DICT = {
    "S01": "가족관계",
    "S02": "학업 및 진로",
    "S03": "학교폭력/따돌림",
    "S04": "대인관계",
    "S05": "연애,결혼,출산",
    "S06": "진로,취업,직장",
    "S07": "대인관계(부부, 자녀)",
    "S08": "재정,은퇴,노후준비",
    "S09": "건강",
    "S10": "직장, 업무 스트레스",
    "S11": "건강,죽음",
    "S12": "대인관계(노년)",
    "S13": "재정",
}

DISEASE_DICT = {
    "D01": "만성질환 유",
    "D02": "만성질환 무",
}

EMOTION_DICT = {
    "E10": "분노",
    "E11": "툴툴대는",
    "E12": "좌절한",
    "E13": "짜증내는",
    "E14": "방어적인",
    "E15": "악의적인",
    "E16": "안달하는",
    "E17": "구역질 나는",
    "E18": "노여워하는",
    "E19": "성가신",
    "E20": "슬픔",
    "E21": "실망한",
    "E22": "비통한",
    "E23": "후회되는",
    "E24": "우울한",
    "E25": "마비된",
    "E26": "염세적인",
    "E27": "눈물이 나는",
    "E28": "낙담한",
    "E29": "환멸을 느끼는",
    "E30": "불안",
    "E31": "두려운",
    "E32": "스트레스 받는",
    "E33": "취약한",
    "E34": "혼란스러운",
    "E35": "당혹스러운",
    "E36": "회의적인",
    "E37": "걱정스러운",
    "E38": "조심스러운",
    "E39": "초조한",
    "E40": "상처",
    "E41": "질투하는",
    "E42": "배신당한",
    "E43": "고립된",
    "E44": "충격 받은",
    "E45": "가난한, 불우한",
    "E46": "희생된",
    "E47": "억울한",
    "E48": "괴로워하는",
    "E49": "버려진",
    "E50": "당황",
    "E51": "고립된(당황한)",
    "E52": "남의 시선을 의식하는",
    "E53": "외로운",
    "E54": "열등감",
    "E55": "죄책감의",
    "E56": "부끄러운",
    "E57": "혐오스러운",
    "E58": "한심한",
    "E59": "혼란스러운(당황한)",
    "E60": "기쁨",
    "E61": "감사하는",
    "E62": "신뢰하는",
    "E63": "편안한",
    "E64": "만족스러운",
    "E65": "흥분",
    "E66": "느긋",
    "E67": "안도",
    "E68": "신이 난",
    "E69": "자신하는",
}

AGE_DICT = {"A01": "청소년", "A02": "청년", "A03": "중년", "A04": "노년"}
GENDER_DICT = {"G01": "남성", "G02": "여성"}


class EmotionalDialogueCorpus(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="SFT",
            data_dir="SFT",
            version="1.1.0",
            description="- 용도: 감성대화로 구성된 데이터 " + DESCRIPTION,
        ),
    ]
    DEFAULT_CONFIG_NAME = "SFT"
    DEFAULT_WRITER_BATCH_SIZE = 1000

    def _info(self):
        features = {
            "id": Value("string"),
            "conversations": [
                {
                    "role": Value("string"),
                    "content": Value("string"),
                }
            ],
            "situation": Value("string"),
            "disease": Value("string"),
            "emotion": Value("string"),
            "metadata": {
                "age": Value("string"),
                "gender": Value("string"),
                "computer": Value("string"),
            },
        }

        return DatasetInfo(
            description=self.config.description,
            version=Version(self.config.version),
            features=Features(features),
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
        part_dict = defaultdict(list)
        for part_path in unzip_dir.rglob("*.zip.part*"):
            parh_stem = str(part_path.parent.joinpath(part_path.stem))
            part_dict[parh_stem].append(part_path)

        for dst_path, part_path_ls in part_dict.items():
            with open(dst_path, "wb") as byte_f:
                for part_path in natsorted(part_path_ls):
                    byte_f.write(part_path.read_bytes())
                    os.remove(part_path)

    def _split_generators(self, dl_manager) -> List[SplitGenerator]:  # type: ignore
        def get_zip_path_ls() -> List[Path]:
            cache_dir = Path(dl_manager.download_config.cache_dir)

            unzip_dir = cache_dir.joinpath(_DATANAME)
            tar_file = cache_dir.joinpath(f"{_DATANAME}.tar")

            if not unzip_dir.exists():
                self.aihub_downloader(tar_file)

                with TarFile(tar_file, "r") as mytar:
                    mytar.extractall(unzip_dir)
                self.concat_zip_part(unzip_dir)
                os.remove(tar_file)

            zip_path_ls = list()
            exclude_ls = ["원천데이터"]
            for zip_path in unzip_dir.rglob("*.zip"):
                if any(exclude in zip_path.as_posix() for exclude in exclude_ls):
                    continue
                zip_path_ls.append(zip_path)
            return zip_path_ls

        zip_path_ls = get_zip_path_ls()

        train_zip_path = [
            zip_path for zip_path in zip_path_ls if zip_path.name == "감성대화말뭉치(최종데이터)_Training.zip"
        ][0]
        valid_zip_path = [
            zip_path for zip_path in zip_path_ls if zip_path.name == "감성대화말뭉치(최종데이터)_Validation.zip"
        ][0]
        split_generator_ls = [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "zip_path": train_zip_path,
                    "split": "train",
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "zip_path": valid_zip_path,
                    "split": "validation",
                },
            ),
        ]

        return split_generator_ls

    def _generate_examples(self, zip_path: Path, split: str) -> Generator:
        for idx, data in enumerate(self._generate_example_sft(zip_path, split)):
            yield (idx, data)

    def get_conversations(self, talk_content: dict) -> List[dict]:
        talk_ls = [(u_id, content) for u_id, content in talk_content.items() if content]
        finial_conversations = list()
        conversations_chunk = list(zip_longest(*[iter(talk_ls)] * 2, fillvalue=None))
        for conversations in conversations_chunk:
            user, assistant = conversations[0], conversations[1]
            if user is None or assistant is None:
                continue
            if "HS" not in user[0] or "SS" not in assistant[0]:
                continue
            conversations = [
                {"role": "user", "content": user[1]},
                {"role": "assistant", "content": assistant[1]},
            ]
            finial_conversations.extend(conversations)

        if len(finial_conversations) % 2 != 0:
            return []

        return finial_conversations

    def _generate_example_sft(self, zip_path: Path, split: str) -> Generator:
        zip = ZipFile(zip_path)
        fileinfo = zip.infolist()[0]
        raw_data_ls = json.load(zip.open(fileinfo))
        for raw_data in raw_data_ls:
            talk_content, talk_id = raw_data["talk"]["content"], raw_data["talk"]["id"]
            emotion, persona = raw_data["profile"]["emotion"], raw_data["profile"]["persona"]

            situation, disease, emotion = emotion["emotion-id"].split("_")
            conversations = self.get_conversations(talk_content)

            if not conversations:
                continue

            yield {
                "id": talk_id["talk-id"],
                "conversations": conversations,
                "situation": SITUATION_DICT[situation],
                "disease": DISEASE_DICT[disease],
                "emotion": EMOTION_DICT[emotion],
                "metadata": {
                    "age": AGE_DICT[persona["human"][0]],
                    "gender": GENDER_DICT[persona["human"][1]],
                    "computer": persona["computer"][0],  # EDA 해봤는데 전부 C01 밖에 없다.
                },
            }
