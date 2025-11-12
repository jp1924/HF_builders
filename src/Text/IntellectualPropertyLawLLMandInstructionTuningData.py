import json
import os
from pathlib import Path
from tarfile import TarFile
from typing import Generator, List
from zipfile import ZipFile

import requests
from datasets import (
    BuilderConfig,
    DatasetInfo,
    Features,
    GeneratorBasedBuilder,
    Split,
    SplitGenerator,
    Value,
)
from datasets import logging as ds_logging
from natsort import natsorted
from tqdm import tqdm

_LICENSE = """AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다."""


DATASET_KEY = "71843"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/0.6/{DATASET_KEY}.do"
_HOMEPAGE = f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"  # no lint

_DATANAME = "IntellectualPropertyLawLLMandInstructionTuningData"
DATASET_SIZE = 0.6708  # GB
_DESCRIPTION = """지식재산권법의 정제된 법률데이터 구축과 초거대 AI학습을 위한 Instruction Tuning 데이터 구축을 목표로 하여, 법률분야에서 데이터와 AI기술의 활용을 높여 산업을 활성화시킴과 동시에, 일반인이 법률 정보를 쉽게 이해할 수 있도록 접근성을 향상시킴 
지식재산권법 서비스 혁신) 정제된 지식재산권법 데이터와 AI기술을 활용한 새로운 지식재산권법 서비스와 기술이 개발되어, 법률 분야의 혁신을 이끌어낼 수 있을 것으로 기대되며, 이를 통해 지식재산권법 서비스의 다양성과 효율성이 증가할 것으로 예상됨

(지식재산권법 정보 접근성 향상) 일반인들이 지식재산권법 정보를 이해하고 활용하기 쉬워질 것으로 예상되며, AI기술을 통해 분석된 지식재산권법 데이터는 일상 생활에서 지식재산권법에 대한 지식을 활용하는 것이 용이해질 것으로 예상됨

(지식재산권법 데이터 품질 개선) 본 사업을 통해 지식재산권법 데이터의 품질이 향상될 것으로 예상되므로, 이에 따라 정제된 지식재산권법 데이터는 초거대 AI모델의 정확성을 높이고, 신뢰할 수 있는 법률 서비스를 제공하는데 도움이 것으로 예상됨

법률 문서 작성 지원
- 계약서, 소송 서류 등 전문적인 법률 문서를 보다 쉽고 빠르게 작성할 수 있음
- 초안 작성, 문서 템플릿 추천 등 자동화된 기능 제공
유사 사례(Case) 탐색
- 과거 판례나 유사 사건을 빠르게 찾고 분석
- 판례 비교, 법률적 근거 제시 등에 활용
법률 이슈 체크
- 문서나 상담 과정에서 특정 법적 문제나 이슈를 빠르게 식별
- 잠재적 분쟁 요소, 위법 요소 등을 미리 파악
법률 상담 
- 초기 상담 시 필요한 기본 법률 정보와 유사 사례를 신속하게 제시
- 변호사, 법무팀이 더 복잡한 사안에 집중할 수 있도록 지원
정부부처/공공기관 및 기업활용
- 행정기관이나 공공기관에서 법령 해석, 문서 작성, 이슈 대응 시 업무 효율화
- 로펌, 기업의 내부 법무팀에서 대량 문서 처리 및 케이스 검토 등에 활용"""

doc_class_map = {
    "1": "판결문",
    "2": "법령",
    "3": "심결례",
    "4": "심결문",
    "5": "유권해석",
}

ds_logging.set_verbosity_info()
logger = ds_logging.get_logger("datasets")


class IntellectualPropertyLawLLMandInstructionTuningData(GeneratorBasedBuilder):
    """Essential Medical Knowledge Data from AI Hub."""

    BUILDER_CONFIGS = [
        BuilderConfig(
            name="CORPUS",
            version="1.1.0",
            description="사전학습 목적으로 제작한 데이터" + _DESCRIPTION,
        ),
        BuilderConfig(
            name="SFT",
            version="1.1.0",
            description=_DESCRIPTION,
        ),
        BuilderConfig(
            name="Summarization",
            version="1.1.0",
            description=_DESCRIPTION,
        ),
    ]

    DEFAULT_CONFIG_NAME = "CORPUS"

    def _info(self) -> DatasetInfo:
        if self.config.name == "CORPUS":
            features = {
                "id": Value("string"),
                "sentence": Value("string"),
                "sentence_ls": [Value("string")],
                "metadata": {
                    "doc_class": Value("string"),
                    "doc_id": Value("string"),
                    "title": Value("string"),
                    "doc_type": Value("string"),
                    "issuer": Value("string"),
                    "date": Value("string"),
                },
            }
        elif self.config.name == "SFT":
            features = {
                "id": Value("string"),
                "conversations": [
                    {"role": Value("string"), "content": Value("string")}
                ],
                "system": Value("string"),
                "prompt": Value("string"),
                "answer": Value("string"),
                "passage": Value("string"),
                "metadata": {
                    "doc_class": Value("string"),
                    "doc_id": Value("string"),
                    "title": Value("string"),
                    "doc_type": Value("string"),
                    "issuer": Value("string"),
                    "date": Value("string"),
                    "instruction_case": Value("string"),
                },
            }
        elif self.config.name == "Summarization":
            features = {
                "id": Value("string"),
                "passage": Value("string"),
                "prompt": Value("string"),
                "answer": Value("string"),
                "metadata": {
                    "doc_class": Value("string"),
                    "doc_id": Value("string"),
                    "title": Value("string"),
                    "doc_type": Value("string"),
                    "issuer": Value("string"),
                    "date": Value("string"),
                },
            }

        return DatasetInfo(
            description=self.config.description,
            version=self.config.version,
            features=Features(features),
            homepage=_HOMEPAGE,
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
                raise BaseException(
                    f"Download failed with HTTP status code: {response.status_code}"
                )

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

        train_src_ls = list(
            filter(lambda path: "Training" in path.as_posix(), src_path_ls)
        )
        valid_src_ls = list(
            filter(lambda path: "Validation" in path.as_posix(), src_path_ls)
        )

        split_generator_ls = [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "file_ls": train_src_ls,
                    "split": "train",
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "file_ls": valid_src_ls,
                    "split": "validation",
                },
            ),
        ]

        return split_generator_ls

    def _generate_examples(self, **kwagrs) -> Generator:
        if self.config.name == "CORPUS":
            for idx, data in enumerate(self._corpus_generate_examples(**kwagrs)):
                data["id"] = str(idx)
                yield idx, data
        elif self.config.name == "SFT":
            for idx, data in enumerate(self._sft_generate_examples(**kwagrs)):
                data["id"] = str(idx)
                yield idx, data
        elif self.config.name == "Summarization":
            for idx, data in enumerate(self._summarization_generate_examples(**kwagrs)):
                data["id"] = str(idx)
                yield idx, data

    def _corpus_generate_examples(self, file_ls: List[Path], split: str) -> Generator:
        def _normalize_metadata(source: dict) -> dict:
            from dateutil import parser as _p

            def first(*keys):
                for k in keys:
                    v = source.get(k)
                    if v not in (None, "", []):
                        return v
                return None

            # 날짜 우선순위 (간단화)
            date_keys = [
                "announce_date",
                "decision_date",
                "response_date",
                "effective_date",
                "proclamation_date",
            ]
            date = None
            for k in date_keys:
                if source.get(k):
                    try:
                        date = _p.parse(source.get(k)).date().isoformat()
                    except Exception:
                        date = str(source.get(k))
                    break

            metadata = {
                "doc_class": doc_class_map.get(str(source.get("doc_class")), "기타")
                if source.get("doc_class")
                else None,
                "doc_id": first("doc_id", "id"),
                "title": first("title", "statute_name", "casenames"),
                "doc_type": first("document_type", "statute_type", "data_class"),
                "issuer": first("normalized_court", "response_institute"),
                "date": date,
            }

            return metadata

        from kss import Kss

        ko_sentence_splitter = Kss("split_sentences")

        src_path_ls = filter(lambda path: "원천데이터" in path.as_posix(), file_ls)
        src_path_ls = natsorted(src_path_ls, key=lambda src_path: src_path.stem)

        for src_path in src_path_ls:
            src_zip = ZipFile(src_path)
            for src_info in src_zip.infolist():
                if src_info.is_dir():
                    continue
                source = json.load(src_zip.open(src_info))

                if isinstance(source["sentences"], str):
                    sentence = source["sentences"].strip()
                    sentence_ls = ko_sentence_splitter(
                        source["sentences"], backend="fast"
                    )
                else:
                    sentence = "\n".join(source["sentences"]).strip()
                    sentence_ls = source["sentences"]

                yield {
                    "id": "",
                    "sentence": sentence,
                    "sentence_ls": sentence_ls,
                    "metadata": _normalize_metadata(source),
                }

    def _sft_generate_examples(self, file_ls: List[Path], split: str) -> Generator:
        def normalize_metadata_sft(source: dict, taskinfo: dict | None = None) -> dict:
            from dateutil import parser as _p

            def first(*keys):
                for k in keys:
                    v = source.get(k)
                    if v not in (None, "", []):
                        return v
                return None

            date_keys = [
                "announce_date",
                "decision_date",
                "response_date",
                "effective_date",
                "proclamation_date",
            ]
            date = None
            for k in date_keys:
                if source.get(k):
                    try:
                        date = _p.parse(source.get(k)).date().isoformat()
                    except Exception:
                        date = str(source.get(k))
                    break

            md = {
                "doc_class": doc_class_map.get(str(source.get("doc_class")), "기타")
                if source.get("doc_class") is not None
                else None,
                "doc_id": first("doc_id", "id"),
                "title": first("title", "statute_name", "casenames"),
                "doc_type": first(
                    "document_type", "statute_type", "data_class", "casetype"
                ),
                "issuer": first("normalized_court", "response_institute"),
                "date": date,
            }
            if taskinfo and taskinfo.get("instruction_case"):
                md["instruction_case"] = taskinfo.get("instruction_case")
            # None 또는 빈 값 제거
            return md

        lbl_path_ls = filter(
            lambda path: "라벨링데이터" in str(path) and "질의" in str(path), file_ls
        )
        lbl_path_ls = natsorted(lbl_path_ls, key=lambda lbl_path: lbl_path.stem)

        for lbl_path in lbl_path_ls:
            lbl_zip = ZipFile(lbl_path)
            for lbl_info in lbl_zip.infolist():
                label = json.load(lbl_zip.open(lbl_info))

                passage = "\n".join(label["taskinfo"]["sentences"]).strip()

                yield {
                    "id": "",
                    "conversations": [
                        {"role": "system", "content": label["taskinfo"]["instruction"]},
                        {"role": "passage", "content": passage},
                        {"role": "user", "content": label["taskinfo"]["input"]},
                        {"role": "assistant", "content": label["taskinfo"]["output"]},
                    ],
                    "system": label["taskinfo"]["instruction"],
                    "prompt": label["taskinfo"]["input"],
                    "answer": label["taskinfo"]["output"],
                    "passage": passage,
                    "metadata": normalize_metadata_sft(
                        label["info"], label["taskinfo"]
                    ),
                }

    def _summarization_generate_examples(
        self, file_ls: List[Path], split: str
    ) -> Generator:
        def _normalize_metadata(source: dict) -> dict:
            from dateutil import parser as _p

            def first(*keys):
                for k in keys:
                    v = source.get(k)
                    if v not in (None, "", []):
                        return v
                return None

            # 날짜 우선순위(간단화)
            date_keys = [
                "announce_date",
                "decision_date",
                "response_date",
                "effective_date",
                "proclamation_date",
            ]
            date = None
            for k in date_keys:
                if source.get(k):
                    try:
                        date = _p.parse(source.get(k)).date().isoformat()
                    except Exception:
                        date = str(source.get(k))
                    break

            metadata = {
                "doc_class": doc_class_map.get(str(source.get("doc_class")), "기타")
                if source.get("doc_class")
                else None,
                "doc_id": first("doc_id", "id"),
                "title": first("title", "statute_name", "casenames"),
                "doc_type": first(
                    "document_type", "statute_type", "data_class", "casetype"
                ),
                "issuer": first("normalized_court", "response_institute"),
                "date": date,
            }

            return metadata

        lbl_path_ls = filter(
            lambda path: "라벨링데이터" in str(path) and "요약" in str(path), file_ls
        )
        lbl_path_ls = natsorted(lbl_path_ls, key=lambda lbl_path: lbl_path.stem)

        for lbl_path in lbl_path_ls:
            lbl_zip = ZipFile(lbl_path)
            for lbl_info in lbl_zip.infolist():
                label = json.load(lbl_zip.open(lbl_info))
                yield {
                    "id": "",
                    "passage": "\n".join(label["taskinfo"]["sentences"]).strip(),
                    "prompt": label["taskinfo"]["instruction"],
                    "answer": label["taskinfo"]["output"],
                    "metadata": _normalize_metadata(label["info"]),
                }
