import json
import os
from pathlib import Path
from typing import List

from datasets import BuilderConfig, DatasetInfo, Features, GeneratorBasedBuilder, Split, SplitGenerator, Value


DESCRIPTION = "국립국어원 신문 말뭉치에서 추출한 기사 4,389건을 대상으로 기사에서 추출한 주제 문장과기사를 요약하여 작성한 문장으로 구성된 말뭉치"
HOMEPAGE = "https://kli.korean.go.kr/corpus/main/requestMain.do?tabType=thumb&lang=ko&keyword=%EB%AC%B8%EC%84%9C%20%EC%9A%94%EC%95%BD%20%EB%A7%90%EB%AD%89%EC%B9%98#down"
LICENSE = "https://kli.korean.go.kr/corpus/boards/termsInfo.do"
VERSION = "1.0.0"


class DocumentSummaryCorpus(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [BuilderConfig(name="default", version=VERSION, description=DESCRIPTION)]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        features = Features(
            {
                "document_id": Value("string"),
                "corpus": Value("string"),
                "sentence_ls": [Value("string")],
                "metadata": {"title": Value("string"), "sub_title": Value("string"), "sub_class": Value("string")},
                "summary_sentences": [Value("string")],
                "category": Value("string"),
            }
        )

        self.file_path = os.getenv("ZIP_FILE_PATH", None)
        if not self.file_path:
            raise ValueError(
                "DocumentSummaryCorpus의 압축 파일 경로가 지정되지 않았습니다."
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
        dst_file_path = dl_manager.download_and_extract(self.file_path)
        json_path_ls = list(Path(dst_file_path).rglob("*.json"))

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
            document_ls = json.loads(posix_path.read_text("utf-8"))["data"]
            for document in document_ls:
                corpus = " ".join(document["topic_sentences"])

                data = {
                    "document_id": document["document_id"],
                    "corpus": corpus,
                    "sentence_ls": document["topic_sentences"],
                    "metadata": {
                        "title": document["head"],
                        "sub_title": document["subhead"],
                        "sub_class": document["subclass"],
                    },
                    "summary_sentences": document["summary_sentences"],
                    "category": None,
                }

                yield (idx, data)
                idx += 1
