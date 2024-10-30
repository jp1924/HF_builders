import json
import os
import re
from pathlib import Path
from typing import List

from datasets import BuilderConfig, DatasetInfo, Features, GeneratorBasedBuilder, Split, SplitGenerator, Value
from kss import Kss


DESCRIPTION = "2020년 일상 대화 말뭉치 구축(2020)"
HOMEPAGE = "https://kli.korean.go.kr/corpus/main/requestMain.do?tabType=thumb&lang=ko&keyword=%EC%9D%BC%EC%83%81%20%EB%8C%80%ED%99%94%20%EB%A7%90%EB%AD%89%EC%B9%98%202020#down"
LICENSE = "https://kli.korean.go.kr/corpus/boards/termsInfo.do"
VERSION = "1.4.0"

noise_regex = re.compile(r"{laughing}|{clearing}|{singing}|{applauding}|\(\(xx\)\)|\(\(|\)\)|~")
incomplete_regex = re.compile(r"-[ㅏ-ㅣ가-힣ㄱ-ㅎ]-")
"""
※ 전사 기호
  - 웃음 {laughing}
  - 목청 가다듬는 소리 {clearing}
  - 노래 {singing}
  - 박수 {applauding}
  - 잘 들리지 않는 부분 ((추정 전사))
  - 들리지 않는 음절 ((xx))
  - 전혀 들리지 않는 부분 (())
  - 담화 표지 ~
  - 불완전 발화 -­불완전 발화-
※ 비식별화 기호
  - 이름 &name&
  - 주민 등록 번호 &social-security-num&
  - 카드 번호 &card-num&
  - 주소 &address&
  - 전화번호 &tel-num&
  - 상호명 &company-name&
"""


class EverydayConversationCorpus2020(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [BuilderConfig(name="default", version=VERSION, description=DESCRIPTION)]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        features = Features(
            {
                "document_id": Value("string"),
                "corpus": Value("string"),
                "sentence_ls": [Value("string")],
                "metadata": {
                    "title": Value("string"),
                    "author": Value("string"),
                    "publisher": Value("string"),
                    "date": Value("string"),
                    "topic": Value("string"),
                    "speaker": {
                        "id": Value("string"),
                        "age": Value("string"),
                        "occupation": Value("string"),
                        "sex": Value("string"),
                        "birthplace": Value("string"),
                        "principal_residence": Value("string"),
                        "current_residence": Value("string"),
                        "education": Value("string"),
                    },
                    "setting": {"relation": Value("string")},
                    "start": Value("float32"),
                    "end": Value("float32"),
                },
                "category": Value("string"),
            }
        )

        self.file_path = os.getenv("ZIP_FILE_PATH", None)
        if not self.file_path:
            raise ValueError(
                f"{self.__class__.__name__}의 압축 파일 경로가 지정되지 않았습니다."
                "`os.environ['ZIP_FILE_PATH']='your_file_path'`으로 압축파일 경로를 지정해 주세요."
            )

        return DatasetInfo(
            description=DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=HOMEPAGE,
            license=LICENSE,
            version=VERSION,
        )

    def _split_generators(self, dl_manager) -> List[SplitGenerator]:  # type: ignore
        json_path_ls = list(Path(self.file_path).rglob("*.json"))

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "filepath": json_path_ls,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: List[Path], split: str):
        split_sentences = Kss("split_sentences")
        idx = 0
        for posix_path in filepath:
            document_ls = json.loads(posix_path.read_text("utf-8"))["document"]
            for document in document_ls:
                utterance_ls = document["utterance"]
                corpus = " ".join([utterance["original_form"] for utterance in utterance_ls])

                corpus = noise_regex.sub("", corpus)
                corpus = incomplete_regex.sub("", corpus)

                breakpoint()

                data = {
                    "document_id": document["id"],
                    "corpus": corpus,
                    "sentence_ls": split_sentences(corpus),
                    "metadata": document["metadata"],
                    "category": None,
                }

                yield (idx, data)
                idx += 1
