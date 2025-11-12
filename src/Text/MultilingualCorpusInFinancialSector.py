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
    Translation,
    Value,
)
from datasets import logging as ds_logging
from natsort import natsorted
from tqdm import tqdm

_LICENSE = """AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다."""


DATASET_KEY = "71782"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/0.5/{DATASET_KEY}.do"
_HOMEPAGE = f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"  # nolint


_DATANAME = "MultilingualCorpusInFinancialSector"
DATASET_SIZE = 17.01  # GB


_DESCRIPTION = """- 국내 금융기관의 금융 분야 통번역 서비스를 지원하기 위한 금융 분야의 다국어 번역 데이터
- 금융 분야의 다국어 번역 병렬 말뭉치를 구축하여, 금융 분야 기계 번역기의 품질 고도화
(한국어↔영어/한국어↔중국어/한국어↔일본어/한국어↔베트남어/한국어↔인도네시아어)

# 데이터 구조
```json
{
    "meta": {
        "doc_no": "paper199726",
        "domain": "금융",
        "category": "학술논문",
        "license": "open",
        "source_language": "ko",
        "target_language": "id",
    },
    "doc_info": {
        "source": "한국재무관리학회",
        "journal_name": "재무관리연구",
        "vol_info": "2021, vol.38,no.1, pp. 79-102 (24 pages)",
        "doi": "10.22510/kjofm.2021.38.1.004",
        "title": "주채무계열 제도가 기업가치에 미치는 영향",
        "date": 202103,
    },
    "sents": [
        {
            "page": 5,
            "sn": "paper199726sent2115084",
            "source_original": "2014년 주채무계열 제도 개선의 일환으로 주채무계열 편입 관련 신용공여액 기준을 낮춤에 따라 대상 기업 수가 일시적으로 증가하였다가 2015년부터 다시 감소세로 돌아서는 것을 볼 수 있다.",
            "source_cleaned": "2014년 주채무계열 제도 개선의 일환으로 주채무계열 편입 관련 신용공여액 기준을 낮춤에 따라 대상 기업 수가 일시적으로 증가하였다가 2015년부터 다시 감소세로 돌아서는 것을 볼 수 있다.",
            "mt": "Sebagai bagian dari perbaikan sistem rantai hutang utama pada tahun 2014, jumlah perusahaan target telah meningkat sementara karena standar pemberian kredit yang terkait dengan transfer hutang utama telah menurun, dan kemudian kembali ke penurunan sejak tahun 2015.",
            "mtpe": "Sebagai bagian dari perbaikan sistem seri utang utama pada tahun 2014, standar pemberian kredit terkait penyertaan dalam seri utang utama diturunkan, dan jumlah perusahaan sasaran untuk sementara bertambah, namun terlihat mulai menurun kembali sejak tahun 2015.",
        },
    ],
}
```
doc_info는 분야에 따라 조건적으로 있을 수 있고, 없을 수 있음. 뭐 뉴스는 doc_info가 없고 이런식임"""


ds_logging.set_verbosity_info()
logger = ds_logging.get_logger("datasets")

lang_map = {
    "en": "Eng",
    "ja": "Jpn",
    "zh": "Cmn",
    "vi": "Vie",
    "id": "Ind",
}
corpus_map = {
    "Eng-Corpus": "영어",
    "Jpn-Corpus": "일본어",
    "Vie-Corpus": "베트남어",
    "Ind-Corpus": "인도네시아어",
    "Cmn-Corpus": "중국어간체",
    "Kor-Corpus": "한국어",
}


# 언어 표현은 639-3으로 표기
class MultilingualCorpusInFinancialSector(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="Original",
            data_dir="Original",
            version="1.2.0",
            description=_DESCRIPTION,
        ),
        BuilderConfig(
            name="Kor-Corpus",
            data_dir="Kor-Corpus",
            version="1.2.0",
            description=_DESCRIPTION,
        ),
        BuilderConfig(
            name="Eng-Corpus",
            data_dir="Eng-Corpus",
            version="1.2.0",
            description=_DESCRIPTION,
        ),
        BuilderConfig(
            name="Jpn-Corpus",
            data_dir="Jpn-Corpus",
            version="1.2.0",
            description=_DESCRIPTION,
        ),
        BuilderConfig(
            name="Vie-Corpus",
            data_dir="Vie-Corpus",
            version="1.2.0",
            description=_DESCRIPTION,
        ),
        BuilderConfig(
            name="Ind-Corpus",
            data_dir="Ind-Corpus",
            version="1.2.0",
            description=_DESCRIPTION,
        ),
        BuilderConfig(
            name="Cmn-Corpus",
            data_dir="Cmn-Corpus",
            version="1.2.0",
            description=_DESCRIPTION,
        ),
        BuilderConfig(
            name="Translation",
            data_dir="Translation",
            version="1.2.0",
            description=_DESCRIPTION,
        ),
    ]
    DEFAULT_CONFIG_NAME = "Translation"

    def _info(self):
        if self.config.name == "Translation":
            features = Features(
                {
                    "id": Value("string"),
                    "translation": Translation(
                        languages=["Kor", "Eng", "Jpn", "Vie", "Ind", "Cmn"]
                    ),
                    "metadata": {
                        "doc_info": {
                            "source": Value("string"),
                            "journal_name": Value("string"),
                            "vol_info": Value("string"),
                            "doi": Value("string"),
                            "title": Value("string"),
                            "date": Value("int32"),
                        },
                        "doc_no": Value("string"),
                        "domain": Value("string"),
                        "category": Value("string"),
                        "license": Value("string"),
                        "source_language": Value("string"),
                        "target_language": Value("string"),
                    },
                }
            )
        elif self.config.name in [
            "Kor-Corpus",
            "Eng-Corpus",
            "Jpn-Corpus",
            "Vie-Corpus",
            "Ind-Corpus",
            "Cmn-Corpus",
        ]:
            features = Features(
                {
                    "id": Value("string"),
                    "corpus": Value("string"),
                    "sentence_ls": [Value("string")],
                    "category": Value("string"),
                    "title": Value("string"),
                }
            )
        elif self.config.name == "Original":
            features = Features(
                {
                    "meta": {
                        "doc_no": Value(dtype="string", id=None),
                        "domain": Value(dtype="string", id=None),
                        "category": Value(dtype="string", id=None),
                        "license": Value(dtype="string", id=None),
                        "source_language": Value(dtype="string", id=None),
                        "target_language": Value(dtype="string", id=None),
                    },
                    "doc_info": {
                        "source": Value(dtype="string", id=None),
                        "journal_name": Value(dtype="string", id=None),
                        "vol_info": Value(dtype="string", id=None),
                        "doi": Value(dtype="string", id=None),
                        "title": Value(dtype="string", id=None),
                        "date": Value(dtype="int64", id=None),
                    },
                    "sents": [
                        {
                            "page": Value(dtype="int64", id=None),
                            "sn": Value(dtype="string", id=None),
                            "source_original": Value(dtype="string", id=None),
                            "source_cleaned": Value(dtype="string", id=None),
                            "mt": Value(dtype="string", id=None),
                            "mtpe": Value(dtype="string", id=None),
                        }
                    ],
                }
            )

        return DatasetInfo(
            description=self.config.description,
            version=self.config.version,
            features=features,
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

        train_src_ls = [path for path in src_path_ls if "Training" in path.as_posix()]
        valid_src_ls = [path for path in src_path_ls if "Validation" in path.as_posix()]

        if self.config.name == "Translation":
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
        elif self.config.name in [
            "Kor-Corpus",
            "Eng-Corpus",
            "Jpn-Corpus",
            "Vie-Corpus",
            "Ind-Corpus",
            "Cmn-Corpus",
        ]:
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
        elif self.config.name == "Original":
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
        else:
            raise ValueError(f"지원하지 않는 config name입니다. {self.config.name}")

        return split_generator_ls

    def _generate_examples(self, **kwagrs) -> Generator:
        if self.config.name == "Translation":
            for idx, data in enumerate(self._translation_generate_examples(**kwagrs)):
                yield idx, data
        elif self.config.name in [
            "Kor-Corpus",
            "Eng-Corpus",
            "Jpn-Corpus",
            "Vie-Corpus",
            "Ind-Corpus",
            "Cmn-Corpus",
        ]:
            for idx, data in enumerate(self._corpus_generate_examples(**kwagrs)):
                yield idx, data
        elif self.config.name == "Original":
            for idx, data in enumerate(self._original_generate_examples(**kwagrs)):
                yield idx, data

    def _translation_generate_examples(
        self,
        file_ls: List[Path],
        split: str,
    ) -> Generator:
        lbl_zip_ls = [
            ZipFile(path) for path in file_ls if "라벨링데이터" in path.as_posix()
        ]

        for lbl_zip in lbl_zip_ls:
            lbl_info_ls = [
                info
                for info in lbl_zip.infolist()
                if not info.is_dir() and "json" in info.filename
            ]
            for lbl_info in lbl_info_ls:
                labels = json.load(lbl_zip.open(lbl_info))
                metadata = {
                    "doc_info": (labels["doc_info"] if "doc_info" in labels else {}),
                    **labels["meta"],
                }

                for sents in labels["sents"]:
                    src_lang, trg_lang = "Kor", lang_map[metadata["target_language"]]

                    translation = {
                        lang: None for lang in lang_map.values() if lang != trg_lang
                    }
                    translation = {
                        **{
                            src_lang: sents["source_original"],
                            trg_lang: sents["mt"],
                        },
                        **translation,
                    }
                    yield {
                        "id": f"{metadata['doc_no']}-{sents['sn']}",
                        "translation": translation,
                        "metadata": metadata,
                    }

    def _corpus_generate_examples(
        self,
        file_ls: List[Path],
        split: str,
    ) -> Generator:
        lbl_zip_ls = natsorted(
            [ZipFile(path) for path in file_ls if "라벨링데이터" in path.as_posix()],
            key=lambda x: Path(x.filename).stem,
        )
        # 맨 마지막은 답변할 수 없는 지문들로만 구성된 데이터.
        corpus_map[self.config.name]

        for lbl_zip in lbl_zip_ls:
            corpus_type = corpus_map[self.config.name]
            if corpus_type != "한국어" and (
                corpus_type not in Path(lbl_zip.filename).stem
            ):
                continue

            lbl_info_ls = natsorted(
                [
                    info
                    for info in lbl_zip.infolist()
                    if not info.is_dir() and "json" in info.filename
                ],
                key=lambda x: x.filename,
            )

            for label_info in lbl_info_ls:
                labels = json.loads(lbl_zip.open(label_info).read())

                sentence_ls = list()
                for sents in labels["sents"]:
                    corpus = (
                        sents["source_cleaned"]
                        if self.config.name == "Kor-Corpus"
                        else sents["mtpe"]
                    )
                    sentence_ls.append(corpus)

                title = None
                if "doc_info" in labels:
                    if "doc_name" in labels["doc_info"]:
                        title = labels["doc_info"]["doc_name"]
                    elif "title" in labels["doc_info"]:
                        title = labels["doc_info"]["title"]
                    else:
                        breakpoint()
                        labels

                yield {
                    "id": f"{labels['meta']['doc_no']}-{sents['sn']}",
                    "corpus": "\n".join(sentence_ls),
                    "sentence_ls": sentence_ls,
                    "category": labels["meta"]["category"],
                    "title": title,
                }

    def _original_generate_examples(self, file_ls: List[Path], split: str) -> Generator:
        lbl_zip_ls = natsorted(
            [ZipFile(path) for path in file_ls if "라벨링데이터" in path.as_posix()],
            key=lambda x: Path(x.filename).stem,
        )
        # 맨 마지막은 답변할 수 없는 지문들로만 구성된 데이터.
        label_zip = lbl_zip_ls[0]
        label_info_ls = natsorted(
            [
                info
                for info in label_zip.infolist()
                if not info.is_dir() and "json" in info.filename
            ],
            key=lambda x: x.filename,
        )
        for label_info in label_info_ls:
            labels = json.loads(label_zip.open(label_info).read())
            yield labels
