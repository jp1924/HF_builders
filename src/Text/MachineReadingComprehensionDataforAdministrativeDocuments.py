import json
import os
import random
from collections import defaultdict
from pathlib import Path
from tarfile import TarFile
from typing import Dict, Generator, List
from zipfile import ZipFile

import requests
from datasets import BuilderConfig, DatasetInfo, Features, GeneratorBasedBuilder, Split, SplitGenerator, Value
from datasets import logging as ds_logging
from natsort import natsorted
from tqdm import tqdm


_LICENSE = """AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다."""


DATASET_KEY = "569"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/0.6/{DATASET_KEY}.do"
_HOMEPAGE = f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"  # no lint


_DATANAME = "MachineReadingComprehensionDataforAdministrativeDocuments"
DATASET_SIZE = 0.2531  # GB


_DESCRIPTION = """행정문서를 활용하여 기계독해 모델 생성을 위한 지문-질문-답변으로 구성된 인공지능 학습 데이터"""

ds_logging.set_verbosity_info()
logger = ds_logging.get_logger("datasets")


class MachineReadingComprehensionDataforAdministrativeDocuments(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(name="unanswerable", version="1.0.0", description=_DESCRIPTION),
        BuilderConfig(name="text_entailment", version="1.0.0", description=_DESCRIPTION),
        BuilderConfig(name="tableqa", version="1.0.0", description=_DESCRIPTION),
        BuilderConfig(name="multiple_choice", version="1.0.0", description=_DESCRIPTION),
        BuilderConfig(name="span_extraction_how", version="1.0.0", description=_DESCRIPTION),
        BuilderConfig(name="span_extraction", version="1.0.0", description=_DESCRIPTION),
    ]
    DEFAULT_CONFIG_NAME = "unanswerable"

    def _info(self):
        if self.config.name == "unanswerable":
            features = Features(
                {
                    "id": Value("string"),
                    "passage": Value("string"),
                    "prompt": Value("string"),
                    "answer": Value("string"),
                    "reason": Value("string"),
                    "answerable": Value("bool"),
                    "answer_idx": Value("int32"),
                    "reason_idx": Value("int32"),
                    "passage_ls": [Value("string")],
                    "metadata": {
                        "doc_id": Value("string"),
                        "doc_title": Value("string"),
                        "doc_source": Value("string"),
                        "doc_published": Value("string"),
                        "doc_class": {"code": Value("string"), "name": Value("string")},
                        "created": Value("string"),
                        "context_id": Value("string"),
                        "question_id": Value("string"),
                        "qa_type": Value("string"),
                        "options": [Value("string")],
                    },
                }
            )
        elif self.config.name == "text_entailment":
            features = Features(
                {
                    "id": Value("string"),
                    "passage": Value("string"),
                    "prompt": Value("string"),
                    "answer": Value("string"),
                    "reason": Value("string"),
                    "answerable": Value("bool"),
                    "answer_idx": Value("int32"),
                    "reason_idx": Value("int32"),
                    "passage_ls": [Value("string")],
                    "metadata": {
                        "doc_id": Value("string"),
                        "doc_title": Value("string"),
                        "doc_source": Value("string"),
                        "doc_published": Value("string"),
                        "doc_class": {"code": Value("string"), "name": Value("string")},
                        "created": Value("string"),
                        "context_id": Value("string"),
                        "question_id": Value("string"),
                        "qa_type": Value("string"),
                        "options": [Value("string")],
                    },
                }
            )
        elif self.config.name == "span_extraction":
            features = Features(
                {
                    "id": Value("string"),
                    "passage": Value("string"),
                    "prompt": Value("string"),
                    "answer": Value("string"),
                    "reason": Value("string"),
                    "answerable": Value("bool"),
                    "answer_idx": Value("int32"),
                    "reason_idx": Value("int32"),
                    "passage_ls": [Value("string")],
                    "metadata": {
                        "doc_id": Value("string"),
                        "doc_title": Value("string"),
                        "doc_source": Value("string"),
                        "doc_published": Value("string"),
                        "doc_class": {"code": Value("string"), "name": Value("string")},
                        "created": Value("string"),
                        "context_id": Value("string"),
                        "question_id": Value("string"),
                        "qa_type": Value("string"),
                        "options": [Value("string")],
                    },
                }
            )
        elif self.config.name == "span_extraction_how":
            features = Features(
                {
                    "id": Value("string"),
                    "passage": Value("string"),
                    "prompt": Value("string"),
                    "answer": Value("string"),
                    "reason": Value("string"),
                    "answerable": Value("bool"),
                    "answer_idx": Value("int32"),
                    "reason_idx": Value("int32"),
                    "passage_ls": [Value("string")],
                    "metadata": {
                        "doc_id": Value("string"),
                        "doc_title": Value("string"),
                        "doc_source": Value("string"),
                        "doc_published": Value("string"),
                        "doc_class": {"code": Value("string"), "name": Value("string")},
                        "created": Value("string"),
                        "context_id": Value("string"),
                        "question_id": Value("string"),
                        "qa_type": Value("string"),
                        "options": [Value("string")],
                    },
                }
            )
        elif self.config.name == "tableqa":
            features = Features(
                {
                    "id": Value("string"),
                    "passage": Value("string"),
                    "prompt": Value("string"),
                    "answer": Value("string"),
                    "reason": Value("string"),
                    "answerable": Value("bool"),
                    "answer_idx": Value("int32"),
                    "reason_idx": Value("int32"),
                    "passage_ls": [Value("string")],
                    "metadata": {
                        "doc_id": Value("string"),
                        "doc_title": Value("string"),
                        "doc_source": Value("string"),
                        "doc_published": Value("string"),
                        "doc_class": {"code": Value("string"), "name": Value("string")},
                        "created": Value("string"),
                        "context_id": Value("string"),
                        "question_id": Value("string"),
                        "qa_type": Value("string"),
                        "options": [Value("string")],
                    },
                }
            )
        elif self.config.name == "multiple_choice":
            features = Features(
                {
                    "id": Value("string"),
                    "passage": Value("string"),
                    "prompt": Value("string"),
                    "answer": Value("string"),
                    "reason": Value("string"),
                    "answerable": Value("bool"),
                    "answer_idx": Value("int32"),
                    "reason_idx": Value("int32"),
                    "choice_ls": [Value("string")],
                    "passage_ls": [Value("string")],
                    "metadata": {
                        "doc_id": Value("string"),
                        "doc_title": Value("string"),
                        "doc_source": Value("string"),
                        "doc_published": Value("string"),
                        "doc_class": {"code": Value("string"), "name": Value("string")},
                        "created": Value("string"),
                        "context_id": Value("string"),
                        "question_id": Value("string"),
                        "qa_type": Value("string"),
                    },
                }
            )

        return DatasetInfo(
            description=self.config.description,
            version=self.config.version,
            homepage=_HOMEPAGE,
            features=features,
            license=_LICENSE,
        )

    def _aihub_downloader(self, download_path: Path) -> List[Path]:
        def download_from_aihub(download_path: Path, apikey: str) -> None:
            # 이유는 모르겠는데 try를 두번 겹쳐야지 정상 동작하더라.
            try:
                try:
                    with TarFile.open(download_path, "r") as tar:
                        tar.getmembers()
                        return None
                except Exception as e:
                    msg = f"tar 파일이 손상되었다. {e} 손상된 파일은 삭제하고 다시 다운로드 받는다."
                    logger.warning(msg)
                    download_path.unlink()
            except BaseException:
                pass

            headers, params = {"apikey": apikey}, {"fileSn": "all"}
            response = requests.get(
                DOWNLOAD_URL,
                headers=headers,
                params=params,
                stream=True,
            )

            if response.status_code == 502:
                raise BaseException(
                    "다운로드 서비스는 홈페이지(https://aihub.or.kr)에서 신청 및 승인 후 이용 가능 합니다."
                )
            if response.status_code != 200:
                raise BaseException(f"Download failed with HTTP status code: {response.status_code}")

            logger.info("다운로드 시작!")
            downloaded_bytes = 0
            data_file = open(download_path.as_posix(), "wb")
            with tqdm(total=round(DATASET_SIZE * 1024**2)) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    data_file.write(chunk)
                    downloaded_bytes += len(chunk)

                    pbar.update(1)
                    prefix = f"Downloaded (GB): {downloaded_bytes / (1024**3):.4f}/{DATASET_SIZE}"
                    pbar.set_postfix_str(prefix)

            data_file.close()

        def concat_zip_part(data_dir: Path) -> None:
            """데이터
            ┣ dataset_0.zip.part0
            ┣ dataset_1.zip.part0
            ┣ dataset_1.zip.part1073741824
            ┣ dataset_1.zip.part10737418240
            ┣ dataset_1.zip.part11811160064
            ┣ dataset_1.zip.part12884901888
            ┣ dataset_1.zip.part13958643712
            ┣ dataset_1.zip.part2147483648
            AI-HUB에서 다운받는 데이터는 part로 나뉘어져 있어서 병합할 필요가 있다."""
            part_dict = dict()
            for part_path in Path(data_dir).rglob("*.part*"):
                parh_stem = str(part_path.parent.joinpath(part_path.stem))
                part_dict.setdefault(parh_stem, list()).append(part_path)

            for dst_path, part_path_ls in part_dict.items():
                with open(dst_path, "wb") as byte_f:
                    for part_path in natsorted(part_path_ls):
                        byte_f.write(part_path.read_bytes())
                        os.remove(part_path)

        def unzip_tar_file(tar_file: Path, unzip_dir: Path) -> None:
            with TarFile(tar_file, "r") as tar:
                tar.extractall(unzip_dir)

            os.remove(tar_file)

        data_dir = download_path.parent.joinpath(download_path.stem)

        complete_file_path = data_dir.joinpath("download_complete")

        if complete_file_path.exists():
            return list(data_dir.rglob("*.zip"))

        aihub_api_key = os.getenv("AIHUB_API_KEY", None)
        if not aihub_api_key:
            raise ValueError(
                """AIHUB_API_KEY가 지정되지 않았습니다. `os.environ["AIHUB_API_KEY"]="your_key"`로 ID를 지정해 주세요"""
            )

        download_from_aihub(download_path, aihub_api_key)
        unzip_tar_file(download_path, data_dir)
        concat_zip_part(data_dir)

        msg = "dataset builder에서 데이터 다시 다운받을지 말지를 결정하는 파일이다. 이거 지우면 aihub에서 데이터 다시 다운 받음."
        complete_file_path.write_text(msg)

        return list(data_dir.rglob("*.zip"))

    def _split_generators(self, dl_manager) -> List[SplitGenerator]:
        cache_dir = Path(dl_manager.download_config.cache_dir)
        download_path = cache_dir.joinpath(f"{_DATANAME}.tar")
        src_path_ls = self._aihub_downloader(download_path)

        train_ls = [p for p in src_path_ls if "Training" in p.as_posix()]
        valid_ls = [p for p in src_path_ls if "Validation" in p.as_posix()]

        suffix_map = {
            "unanswerable": "unanswerable",
            "text_entailment": "text_entailment",
            "tableqa": "tableqa",
            "multiple_choice": "multiple_choice",
            "span_extraction_how": "span_extraction_how",
            "span_extraction": "span_extraction",
        }
        suffix = suffix_map[self.config.name]

        # Training: TS_/TL_, Validation: VS_/VL_
        ts, tl = f"TS_{suffix}", f"TL_{suffix}"
        vs, vl = f"VS_{suffix}", f"VL_{suffix}"

        def _find_first(ls, token):
            return next((p for p in ls if token in p.name), None)

        train_src = _find_first(train_ls, ts)
        train_lbl = _find_first(train_ls, tl)
        valid_src = _find_first(valid_ls, vs)
        valid_lbl = _find_first(valid_ls, vl)

        split_generator_ls = [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"src_path": train_src, "lbl_path": train_lbl}),
            SplitGenerator(name=Split.VALIDATION, gen_kwargs={"src_path": valid_src, "lbl_path": valid_lbl}),
        ]

        return split_generator_ls

    def _generate_examples(self, **kwargs) -> Generator:
        if self.config.name == "unanswerable":
            for idx, data in enumerate(self._unanswerable_generate_examples(**kwargs)):
                yield idx, data
        elif self.config.name == "text_entailment":
            for idx, data in enumerate(self._text_entailment_generate_examples(**kwargs)):
                yield idx, data
        elif self.config.name == "span_extraction":
            for idx, data in enumerate(self._span_extraction_generate_examples(**kwargs)):
                yield idx, data
        elif self.config.name == "span_extraction_how":
            for idx, data in enumerate(self._span_extraction_how_generate_examples(**kwargs)):
                yield idx, data
        elif self.config.name == "tableqa":
            for idx, data in enumerate(self._tableqa_generate_examples(**kwargs)):
                yield idx, data
        elif self.config.name == "multiple_choice":
            for idx, data in enumerate(self._multiple_choice_generate_examples(**kwargs)):
                yield idx, data

    def _collect_passages_by_doc_class(self, labels: Dict) -> Dict[str, List[str]]:
        """doc_class별로 passage를 수집"""
        passages_by_class = defaultdict(list)

        for label in labels["data"]:
            doc_class_code = label["doc_class"]["code"]
            for paragraph in label["paragraphs"]:
                passages_by_class[doc_class_code].append(paragraph["context"])

        return passages_by_class

    def _create_passage_list(
        self,
        positive_passage: str,
        doc_class_code: str,
        passages_by_class: Dict[str, List[str]],
        num_negatives: int = 3,
    ) -> List[str]:
        """positive 1개와 negative passage들로 구성된 리스트 생성"""
        available_passages = [p for p in passages_by_class.get(doc_class_code, []) if p != positive_passage]

        # negative passage 개수 조정 (사용 가능한 passage가 부족할 경우)
        actual_num_negatives = min(num_negatives, len(available_passages))

        if actual_num_negatives == 0:
            return [positive_passage]

        negative_passages = random.sample(available_passages, actual_num_negatives)

        # positive와 negative를 합쳐서 섞기
        passage_list = [positive_passage] + negative_passages
        random.shuffle(passage_list)

        return passage_list

    def _unanswerable_generate_examples(self, src_path: Path, lbl_path: Path) -> Generator:
        lbl_zip = ZipFile(lbl_path, "r")
        labels = json.loads(lbl_zip.read(lbl_zip.namelist()[0]).decode("utf-8"))

        # doc_class별로 passage 수집
        passages_by_class = self._collect_passages_by_doc_class(labels)

        for label in labels["data"]:
            doc_class_code = label["doc_class"]["code"]

            for paragraph in label["paragraphs"]:
                for qa in paragraph["qas"]:
                    # passage_ls 생성 (negative 3개 시도, 부족하면 2개 이하도 가능)
                    passage_ls = self._create_passage_list(
                        positive_passage=paragraph["context"],
                        doc_class_code=doc_class_code,
                        passages_by_class=passages_by_class,
                        num_negatives=3,
                    )

                    yield {
                        "id": f"{label['doc_id']}_{paragraph['context_id']}_{qa['question_id']}",
                        "passage": paragraph["context"],
                        "prompt": qa["question"],
                        "answer": qa["answers"]["text"],
                        "reason": qa["answers"].get("clue_text"),
                        "answerable": not qa["is_impossible"],
                        "answer_idx": qa["answers"].get("answer_start"),
                        "reason_idx": qa["answers"].get("clue_start"),
                        "passage_ls": passage_ls,
                        "metadata": {
                            "doc_id": label["doc_id"],
                            "doc_title": label["doc_title"],
                            "doc_source": label["doc_source"],
                            "doc_published": label["doc_published"],
                            "doc_class": label["doc_class"],
                            "created": label["created"],
                            "context_id": paragraph["context_id"],
                            "question_id": qa["question_id"],
                            "qa_type": qa["qa_type"],
                            "options": qa["answers"].get("options"),
                        },
                    }

    def _text_entailment_generate_examples(self, src_path: Path, lbl_path: Path) -> Generator:
        lbl_zip = ZipFile(lbl_path, "r")
        labels = json.loads(lbl_zip.read(lbl_zip.namelist()[0]).decode("utf-8"))

        # doc_class별로 passage 수집
        passages_by_class = self._collect_passages_by_doc_class(labels)

        for label in labels["data"]:
            doc_class_code = label["doc_class"]["code"]

            for paragraph in label["paragraphs"]:
                for qa in paragraph["qas"]:
                    # passage_ls 생성
                    passage_ls = self._create_passage_list(
                        positive_passage=paragraph["context"],
                        doc_class_code=doc_class_code,
                        passages_by_class=passages_by_class,
                        num_negatives=3,
                    )

                    yield {
                        "id": f"{label['doc_id']}_{paragraph['context_id']}_{qa['question_id']}",
                        "passage": paragraph["context"],
                        "prompt": qa["question"],
                        "answer": qa["answers"]["text"],
                        "reason": qa["answers"].get("clue_text", ""),
                        "answerable": not qa["is_impossible"],
                        "answer_idx": qa["answers"].get("answer_start"),
                        "reason_idx": qa["answers"].get("clue_start"),
                        "passage_ls": passage_ls,
                        "metadata": {
                            "doc_id": label["doc_id"],
                            "doc_title": label["doc_title"],
                            "doc_source": label["doc_source"],
                            "doc_published": label["doc_published"],
                            "doc_class": label["doc_class"],
                            "created": label["created"],
                            "context_id": paragraph["context_id"],
                            "question_id": qa["question_id"],
                            "qa_type": qa["qa_type"],
                            "options": qa["answers"].get("options"),
                        },
                    }

    def _span_extraction_generate_examples(self, src_path: Path, lbl_path: Path) -> Generator:
        lbl_zip = ZipFile(lbl_path, "r")
        labels = json.loads(lbl_zip.read(lbl_zip.namelist()[0]).decode("utf-8"))

        # doc_class별로 passage 수집
        passages_by_class = self._collect_passages_by_doc_class(labels)

        for label in labels["data"]:
            doc_class_code = label["doc_class"]["code"]

            for paragraph in label["paragraphs"]:
                for qa in paragraph["qas"]:
                    # passage_ls 생성
                    passage_ls = self._create_passage_list(
                        positive_passage=paragraph["context"],
                        doc_class_code=doc_class_code,
                        passages_by_class=passages_by_class,
                        num_negatives=3,
                    )

                    yield {
                        "id": f"{label['doc_id']}_{paragraph['context_id']}_{qa['question_id']}",
                        "passage": paragraph["context"],
                        "prompt": qa["question"],
                        "answer": qa["answers"]["text"],
                        "reason": qa["answers"].get("clue_text"),
                        "answerable": not qa["is_impossible"],
                        "answer_idx": qa["answers"]["answer_start"],
                        "reason_idx": qa["answers"].get("clue_start"),
                        "passage_ls": passage_ls,
                        "metadata": {
                            "doc_id": label["doc_id"],
                            "doc_title": label["doc_title"],
                            "doc_source": label["doc_source"],
                            "doc_published": label["doc_published"],
                            "doc_class": label["doc_class"],
                            "created": label["created"],
                            "context_id": paragraph["context_id"],
                            "question_id": qa["question_id"],
                            "qa_type": qa["qa_type"],
                            "options": qa["answers"].get("options"),
                        },
                    }

    def _span_extraction_how_generate_examples(self, src_path: Path, lbl_path: Path) -> Generator:
        lbl_zip = ZipFile(lbl_path, "r")
        labels = json.loads(lbl_zip.read(lbl_zip.namelist()[0]).decode("utf-8"))

        # doc_class별로 passage 수집
        passages_by_class = self._collect_passages_by_doc_class(labels)

        for label in labels["data"]:
            doc_class_code = label["doc_class"]["code"]

            for paragraph in label["paragraphs"]:
                for qa in paragraph["qas"]:
                    # passage_ls 생성
                    passage_ls = self._create_passage_list(
                        positive_passage=paragraph["context"],
                        doc_class_code=doc_class_code,
                        passages_by_class=passages_by_class,
                        num_negatives=3,
                    )

                    yield {
                        "id": f"{label['doc_id']}_{paragraph['context_id']}_{qa['question_id']}",
                        "passage": paragraph["context"],
                        "prompt": qa["question"],
                        "answer": qa["answers"]["text"],
                        "reason": qa["answers"].get("clue_text"),
                        "answerable": not qa["is_impossible"],
                        "answer_idx": qa["answers"]["answer_start"],
                        "reason_idx": qa["answers"].get("clue_start"),
                        "passage_ls": passage_ls,
                        "metadata": {
                            "doc_id": label["doc_id"],
                            "doc_title": label["doc_title"],
                            "doc_source": label["doc_source"],
                            "doc_published": label["doc_published"],
                            "doc_class": label["doc_class"],
                            "created": label["created"],
                            "context_id": paragraph["context_id"],
                            "question_id": qa["question_id"],
                            "qa_type": qa["qa_type"],
                            "options": qa["answers"].get("options"),
                        },
                    }

    def _tableqa_generate_examples(self, src_path: Path, lbl_path: Path) -> Generator:
        lbl_zip = ZipFile(lbl_path, "r")
        labels = json.loads(lbl_zip.read(lbl_zip.namelist()[0]).decode("utf-8"))

        # doc_class별로 passage 수집
        passages_by_class = self._collect_passages_by_doc_class(labels)

        for label in labels["data"]:
            doc_class_code = label["doc_class"]["code"]

            for paragraph in label["paragraphs"]:
                for qa in paragraph["qas"]:
                    # passage_ls 생성
                    passage_ls = self._create_passage_list(
                        positive_passage=paragraph["context"],
                        doc_class_code=doc_class_code,
                        passages_by_class=passages_by_class,
                        num_negatives=3,
                    )

                    yield {
                        "id": f"{label['doc_id']}_{paragraph['context_id']}_{qa['question_id']}",
                        "passage": paragraph["context"],
                        "prompt": qa["question"],
                        "answer": qa["answers"]["text"],
                        "reason": qa["answers"].get("clue_text", ""),
                        "answerable": not qa["is_impossible"],
                        "answer_idx": qa["answers"]["answer_start"],
                        "reason_idx": qa["answers"].get("clue_start"),
                        "passage_ls": passage_ls,
                        "metadata": {
                            "doc_id": label["doc_id"],
                            "doc_title": label["doc_title"],
                            "doc_source": label["doc_source"],
                            "doc_published": label["doc_published"],
                            "doc_class": label["doc_class"],
                            "created": label["created"],
                            "context_id": paragraph["context_id"],
                            "question_id": qa["question_id"],
                            "qa_type": qa["qa_type"],
                            "options": qa["answers"].get("options"),
                        },
                    }

    def _multiple_choice_generate_examples(self, src_path: Path, lbl_path: Path) -> Generator:
        lbl_zip = ZipFile(lbl_path, "r")
        labels = json.loads(lbl_zip.read(lbl_zip.namelist()[0]).decode("utf-8"))

        # doc_class별로 passage 수집
        passages_by_class = self._collect_passages_by_doc_class(labels)

        for label in labels["data"]:
            doc_class_code = label["doc_class"]["code"]

            for paragraph in label["paragraphs"]:
                for qa in paragraph["qas"]:
                    # passage_ls 생성
                    passage_ls = self._create_passage_list(
                        positive_passage=paragraph["context"],
                        doc_class_code=doc_class_code,
                        passages_by_class=passages_by_class,
                        num_negatives=3,
                    )

                    yield {
                        "id": f"{label['doc_id']}_{paragraph['context_id']}_{qa['question_id']}",
                        "passage": paragraph["context"],
                        "prompt": qa["question"],
                        "answer": qa["answers"]["text"],
                        "reason": qa["answers"].get("clue_text", ""),
                        "answerable": not qa["is_impossible"],
                        "answer_idx": qa["answers"].get("answer_start"),
                        "reason_idx": qa["answers"].get("clue_start"),
                        "choice_ls": qa["answers"].get("options", []),
                        "passage_ls": passage_ls,
                        "metadata": {
                            "doc_id": label["doc_id"],
                            "doc_title": label["doc_title"],
                            "doc_source": label["doc_source"],
                            "doc_published": label["doc_published"],
                            "doc_class": label["doc_class"],
                            "created": label["created"],
                            "context_id": paragraph["context_id"],
                            "question_id": qa["question_id"],
                            "qa_type": qa["qa_type"],
                        },
                    }
