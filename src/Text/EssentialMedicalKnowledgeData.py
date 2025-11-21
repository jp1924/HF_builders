import json
import os
from pathlib import Path
from tarfile import TarFile
from typing import Generator, List
from zipfile import ZipFile

import requests
from datasets import BuilderConfig, DatasetInfo, Features, GeneratorBasedBuilder, Split, SplitGenerator, Value
from datasets import logging as ds_logging
from natsort import natsorted
from tqdm import tqdm


_LICENSE = """AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다."""


DATASET_KEY = "71875"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/0.6/{DATASET_KEY}.do"
_HOMEPAGE = f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"  # no lint

_DATANAME = "EssentialMedicalKnowledgeData"
DATASET_SIZE = 0.13612  # GB
_DESCRIPTION = """● 임상학적 근거가 분명한 필수의료 의학지식(질병의 원인, 진단, 치료, 관리, 예방, 최신지견 등)을 내포한 영어/한글 필수의료 의학 말뭉치 1억 토큰 구축/개방
● 향후 의학지식에 대한 전문적 자연어처리 태스크에 대응할 수 있는 필수의료 의학 지식 질의응답 1.5만 쌍 (10만 토큰) 구축/개방● 진료 보조, 예방 및 건강증진, 의료 연구 및 교육에 있어 본 사업의 개방 데이터와 AI 모델은 의료분야 생성형 AI의 근간이 될 것
● 대규모 전문의가 참여한 고품질 데이터셋을 개발하여, 국내 의료분야 인공지능, 나아가 우리나라 인공지능의 국제 경쟁력 향상에 도움이 될 것이라 기대
● 의료자원이 제한된 의료환경에서 저비용 고효율로 다양한 의료분야 애플리케이션의 AI 가속화에 기여하고, 이를 통해 최적의 치료 계획 제공, 건강관리 및 공공보건 관리의 질적 향상, 올바른 정보제공을 통한 일반 국민들의 건강 관련 정보비대칭 해소에 도움이 될 것
1. 데이터 구축 규모
● 원천데이터: 101,400,003 토큰
● 라벨링데이터: 19,201 쌍

2. 데이터 분포 (단위: 쌍)
1. domain
● 산부인과: 2518
● 소아청소년과: 3087
● 응급의학과: 815
● 내과: 12781

2. q_type
● 객관식: 15600
● 단답형: 1814
● 서술형: 1787"""


domain_map = {
    1: "외과",
    2: "예방의학",
    3: "정신건강의학과",
    4: "신경과/신경외과",
    5: "피부과",
    6: "안과",
    7: "이비인후과",
    8: "비뇨의학과",
    9: "방사선종양학과",
    10: "병리과",
    11: "마취통증의학과",
    12: "의료법규",
    13: "기타",
    14: "산부인과",
    15: "소아청소년과",
    16: "응급의학과",
    17: "내과",
}
q_type_map = {
    1: "객관식",
    2: "단답형",
    3: "서술형",
}
qa_id_map = {
    1: "보라매병원",
    2: "삼성서울병원",
    3: "서울대병원",
    4: "서울성모병원",
    5: "세브란스병원",
    6: "크라우드웍스",
    7: "기타",
}

ds_logging.set_verbosity_info()
logger = ds_logging.get_logger("datasets")


class EssentialMedicalKnowledgeData(GeneratorBasedBuilder):
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
            name="GRPO",
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
                "sentence_ls": Value("string"),
                "metadata": {
                    "source_spec": Value("string"),
                    "source": Value("string"),
                    "domain": Value("string"),
                    "creation_year": Value("string"),
                    "language": Value("string"),
                },
            }
        elif self.config.name == "SFT":
            features = {
                "id": Value("string"),
                "prompt": Value("string"),
                "answer": Value("string"),
                "metadata": {
                    "q_type": Value("string"),
                    "domain": Value("string"),
                },
            }
        elif self.config.name == "GRPO":
            features = {
                "id": Value("string"),
                "prompt": Value("string"),
                "answer": Value("string"),
                "metadata": {
                    "q_type": Value("string"),
                    "domain": Value("string"),
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

        train_src_ls = list(filter(lambda path: "Training" in path.as_posix(), src_path_ls))
        valid_src_ls = list(filter(lambda path: "Validation" in path.as_posix(), src_path_ls))

        if self.config.name == "CORPUS":
            split_generator_ls = [
                SplitGenerator(
                    name=Split.TRAIN,
                    gen_kwargs={
                        "file_ls": train_src_ls,
                        "split": "train",
                    },
                )
            ]
        elif self.config.name in ["SFT", "GRPO"]:
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
                yield idx, data
        elif self.config.name == "SFT":
            for idx, data in enumerate(self._sft_generate_examples(**kwagrs)):
                yield idx, data
        elif self.config.name == "GRPO":
            for idx, data in enumerate(self._grpo_generate_examples(**kwagrs)):
                yield idx, data

    def _corpus_generate_examples(self, file_ls: List[Path], split: str) -> Generator:
        import nltk.data
        from kss import Kss

        nltk.download("punkt_tab")
        ko_sentence_splitter = Kss("split_sentences")
        en_sentence_splitter = nltk.data.load("tokenizers/punkt/english.pickle")

        src_path_ls = filter(lambda path: "원천데이터" in path.as_posix(), file_ls)
        src_path_ls = natsorted(src_path_ls, key=lambda src_path: src_path.stem)

        for src_path in src_path_ls:
            src_zip = ZipFile(src_path)
            language = "ko" if "국문" in src_path.as_posix() else "en"  # 영문: en, 국문: ko
            for src_info in src_zip.infolist():
                source = json.load(src_zip.open(src_info))

                if language == "ko":
                    sentence_ls = ko_sentence_splitter(source["content"], backend="fast")
                else:
                    sentence_ls = en_sentence_splitter.tokenize(source["content"])

                yield {
                    "id": str(source["c_id"]),
                    "sentence": source["content"],
                    "sentence_ls": sentence_ls,
                    "metadata": {
                        "source_spec": source["source_spec"],
                        "source": qa_id_map[source["source"]],
                        "domain": domain_map[source["domain"]],
                        "creation_year": source["creation_year"],
                        "language": language,
                    },
                }

    def _sft_generate_examples(self, file_ls: List[Path], split: str) -> Generator:
        lbl_path_ls = filter(lambda path: "라벨링데이터" in str(path), file_ls)
        lbl_path_ls = natsorted(lbl_path_ls, key=lambda lbl_path: lbl_path.stem)

        for lbl_path in lbl_path_ls:
            lbl_zip = ZipFile(lbl_path)
            for lbl_info in lbl_zip.infolist():
                label = json.load(lbl_zip.open(lbl_info))
                yield {
                    "id": str(label["qa_id"]),
                    "prompt": label["question"],
                    "answer": label["answer"],
                    "metadata": {
                        "q_type": q_type_map[label["q_type"]],
                        "domain": domain_map[label["domain"]],
                    },
                }

    def _grpo_generate_examples(self, file_ls: List[Path], split: str) -> Generator:
        lbl_path_ls = filter(lambda path: "라벨링데이터" in str(path), file_ls)
        lbl_path_ls = natsorted(lbl_path_ls, key=lambda lbl_path: lbl_path.stem)

        for lbl_path in lbl_path_ls:
            lbl_zip = ZipFile(lbl_path)
            for lbl_info in lbl_zip.infolist():
                label = json.load(lbl_zip.open(lbl_info))

                if q_type_map[label["q_type"]] not in ["객관식", "단답형"]:
                    continue

                yield {
                    "id": str(label["qa_id"]),
                    "prompt": label["question"],
                    "answer": label["answer"],
                    "metadata": {
                        "q_type": q_type_map[label["q_type"]],
                        "domain": domain_map[label["domain"]],
                    },
                }
