import json
import os
from pathlib import Path
from typing import List

from datasets import BuilderConfig, DatasetInfo, Features, GeneratorBasedBuilder, Split, SplitGenerator, Value


DESCRIPTION = "2022년 신문 기사 원문 자료 수집 및 정제"
HOMEPAGE = "https://kli.korean.go.kr/corpus/main/requestMain.do?tabType=thumb&lang=ko&keyword=%EC%8B%A0%EB%AC%B8%20%EB%A7%90%EB%AD%89%EC%B9%98%202022#down"
LICENSE = "https://kli.korean.go.kr/corpus/boards/termsInfo.do"
VERSION = "1.0.0"


class NewspaperCorpus2021(GeneratorBasedBuilder):
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
                    "original_topic": Value("string"),
                },
                "category": Value("string"),
            }
        )

        self.file_path = os.getenv("ZIP_FILE_PATH", None)
        if not self.file_path:
            raise ValueError(
                "NewspaperCorpus2021의 압축 파일 경로가 지정되지 않았습니다."
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
        idx = 0
        for posix_path in filepath:
            document_ls = json.loads(posix_path.read_text("utf-8"))["document"]
            for document in document_ls:
                sentence_ls = [sentence["form"] for sentence in document["paragraph"]]

                data = {
                    "document_id": document["id"],
                    "corpus": " ".join(sentence_ls),
                    "sentence_ls": sentence_ls,
                    "metadata": document["metadata"],
                    "category": document["metadata"]["topic"],
                }

                yield (idx, data)
                idx += 1
