import json
import os
from collections import Counter
from pathlib import Path
from typing import List

from datasets import BuilderConfig, DatasetInfo, Features, GeneratorBasedBuilder, Split, SplitGenerator, Value
from kss import Kss


DESCRIPTION = "19년 웹 말뭉치에서 선정된 후기 문서 2,081건을 대상으로 작성자의 주관성이 드러나는 표현(감성 표현)을 대상으로 감성 분석 정보를 부착한 말뭉치"
HOMEPAGE = "https://kli.korean.go.kr/corpus/main/requestMain.do?tabType=thumb&lang=ko&keyword=%EA%B0%90%EC%84%B1%20%EB%B6%84%EC%84%9D%20%EB%A7%90%EB%AD%89%EC%B9%98%202020#down"
LICENSE = "https://kli.korean.go.kr/corpus/boards/termsInfo.do"
VERSION = "1.0.0"


class SentimentAnalysisCorpus2020(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [BuilderConfig(name="default", version=VERSION, description=DESCRIPTION)]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        features = Features(
            {
                "document_id": Value("string"),
                "corpus": Value("string"),
                "sentence_ls": [Value("string")],
                "category": Value("string"),
                "metadata": {
                    "title": Value("string"),
                    "author": Value("string"),
                    "publisher": Value("string"),
                    "date": Value("string"),
                    "url": Value("string"),
                },
                "sentiment_expression": [
                    {
                        "expression_id": Value("int32"),
                        "expression": [
                            {
                                "expression_form": Value("string"),
                                "paragraph_id": Value("string"),
                                "begin": Value("int32"),
                                "end": Value("int32"),
                            }
                        ],
                        "expression_score": Value("int32"),
                        "expression_category": [Value("string")],
                        "subject_category": Value("string"),
                        "subject": [Value("string")],
                    }
                ],
            }
        )

        self.file_path = os.getenv("ZIP_FILE_PATH", None)
        if not self.file_path:
            raise ValueError(
                "SentimentAnalysisCorpus의 압축 파일 경로가 지정되지 않았습니다."
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
        split_sentences = Kss("split_sentences")
        idx = 0
        for posix_path in filepath:
            document_ls = json.loads(posix_path.read_text("utf-8"))["document"]
            for document in document_ls:
                corpus = " ".join([paragraph["paragraph_form"] for paragraph in document["paragraph"]])
                counted_obj = Counter([obj["subject_category"] for obj in document["sentiment_expression"]])

                max_count_obj = max(dict(counted_obj).values())
                category = {v: k for k, v in counted_obj.items()}[max_count_obj]

                data = {
                    "document_id": document["document_id"],
                    "corpus": corpus,
                    "sentence_ls": split_sentences(corpus),
                    "metadata": document["metadata"],
                    "sentiment_expression": document["sentiment_expression"],
                    "category": category,
                }

                yield (idx, data)
                idx += 1
