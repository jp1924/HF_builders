import os
from pathlib import Path
from tarfile import TarFile
from typing import Generator, List
from zipfile import ZipFile

import requests
from datasets import (
    Audio,
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


_LICENSE = """
AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다.
"""

DATASET_KEY = "115"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/0.5/{DATASET_KEY}.do"
_HOMEPAGE = "https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=115"
SAMPLE_RATE = 16000
DATASET_SIZE = 360.14

_DATANAME = "KlecSpeech"

DESCRIPTION = """폴더 구조
    └─009.한국어 강의 데이터
        └─01.데이터
            ├─1.Training
            │  ├─라벨링데이터_0908_add
            │  │  ├─KlecSpeech_train_D01_label_0.zip | 39 MB | 48444
            │  │  ├─KlecSpeech_train_D01_label_1.zip | 21 MB | 48445
            │  │  ├─KlecSpeech_train_D01_label_2.zip | 35 MB | 48446
            │  │  ├─KlecSpeech_train_D01_label_3.zip | 9 MB | 48447
            │  │  ├─KlecSpeech_train_D02_label_0.zip | 38 MB | 48448
            │  │  ├─KlecSpeech_train_D02_label_1.zip | 28 MB | 48449
            │  │  ├─KlecSpeech_train_D02_label_2.zip | 32 MB | 48450
            │  │  ├─KlecSpeech_train_D02_label_3.zip | 16 MB | 48451
            │  │  ├─KlecSpeech_train_D02_label_4.zip | 7 MB | 48452
            │  │  ├─KlecSpeech_train_D03_label_0.zip | 26 MB | 48453
            │  │  ├─KlecSpeech_train_D03_label_1.zip | 23 MB | 48454
            │  │  ├─KlecSpeech_train_D03_label_2.zip | 42 MB | 48455
            │  │  ├─KlecSpeech_train_D10_label_0.zip | 8 MB | 48456
            │  │  ├─KlecSpeech_train_D03_label_3.zip | 11 MB | 48372
            │  │  ├─KlecSpeech_train_D04_label_0.zip | 28 MB | 48373
            │  │  ├─KlecSpeech_train_D04_label_1.zip | 23 MB | 48374
            │  │  ├─KlecSpeech_train_D04_label_2.zip | 43 MB | 48375
            │  │  ├─KlecSpeech_train_D11_label_0.zip | 5 MB | 48376
            │  │  ├─KlecSpeech_train_D12_label_0.zip | 6 MB | 48377
            │  │  ├─KlecSpeech_train_D13_label_0.zip | 6 MB | 48378
            │  │  ├─KlecSpeech_train_D14_label_0.zip | 19 MB | 48379
            │  │  ├─KlecSpeech_train_D15_label_0.zip | 5 MB | 48380
            │  │  ├─KlecSpeech_train_D16_label_0.zip | 10 MB | 48381
            │  │  ├─KlecSpeech_train_D17_label_0.zip | 4 MB | 48382
            │  │  ├─KlecSpeech_train_D18_label_0.zip | 8 MB | 48383
            │  │  ├─KlecSpeech_train_D19_label_0.zip | 8 MB | 48384
            │  │  ├─KlecSpeech_train_D99_label_0.zip | 8 MB | 48385
            │  │  ├─KlecSpeech_train_D04_label_3.zip | 8 MB | 48905
            │  │  ├─KlecSpeech_train_D05_label_0.zip | 39 MB | 48865
            │  │  ├─KlecSpeech_train_D05_label_1.zip | 37 MB | 48814
            │  │  ├─KlecSpeech_train_D05_label_2.zip | 17 MB | 48763
            │  │  ├─KlecSpeech_train_D07_label_0.zip | 9 MB | 48690
            │  │  ├─KlecSpeech_train_D08_label_0.zip | 8 MB | 48650
            │  │  ├─KlecSpeech_train_D06_label_0.zip | 9 MB | 48727
            │  │  ├─KlecSpeech_train_D09_label_0.zip | 8 MB | 48483
            │  │  └─KlecSpeech_train_D05_label_3.zip | 12 MB | 48726
            │  └─원천데이터_0908_add
            │      ├─KlecSpeech_train_D02_wav_4.zip | 3 GB | 48394
            │      ├─KlecSpeech_train_D03_wav_0.zip | 15 GB | 48395
            │      ├─KlecSpeech_train_D03_wav_1.zip | 14 GB | 48396
            │      ├─KlecSpeech_train_D03_wav_2.zip | 23 GB | 48397
            │      ├─KlecSpeech_train_D03_wav_3.zip | 5 GB | 48398
            │      ├─KlecSpeech_train_D04_wav_0.zip | 17 GB | 48399
            │      ├─KlecSpeech_train_D04_wav_1.zip | 14 GB | 48400
            │      ├─KlecSpeech_train_D04_wav_2.zip | 22 GB | 48401
            │      ├─KlecSpeech_train_D04_wav_3.zip | 4 GB | 48402
            │      ├─KlecSpeech_train_D05_wav_0.zip | 22 GB | 48403
            │      ├─KlecSpeech_train_D05_wav_1.zip | 19 GB | 48404
            │      ├─KlecSpeech_train_D05_wav_2.zip | 10 GB | 48405
            │      ├─KlecSpeech_train_D15_wav_0.zip | 2 GB | 48416
            │      ├─KlecSpeech_train_D01_wav_0.zip | 22 GB | 48386
            │      ├─KlecSpeech_train_D01_wav_1.zip | 13 GB | 48387
            │      ├─KlecSpeech_train_D01_wav_2.zip | 22 GB | 48388
            │      ├─KlecSpeech_train_D01_wav_3.zip | 4 GB | 48389
            │      ├─KlecSpeech_train_D02_wav_0.zip | 22 GB | 48390
            │      ├─KlecSpeech_train_D02_wav_1.zip | 14 GB | 48391
            │      ├─KlecSpeech_train_D02_wav_2.zip | 18 GB | 48392
            │      ├─KlecSpeech_train_D02_wav_3.zip | 8 GB | 48393
            │      ├─KlecSpeech_train_D14_wav_0.zip | 8 GB | 48415
            │      ├─KlecSpeech_train_D16_wav_0.zip | 4 GB | 48326
            │      ├─KlecSpeech_train_D17_wav_0.zip | 2 GB | 48327
            │      ├─KlecSpeech_train_D18_wav_0.zip | 3 GB | 48328
            │      ├─KlecSpeech_train_D19_wav_0.zip | 4 GB | 48329
            │      ├─KlecSpeech_train_D99_wav_0.zip | 3 GB | 48330
            │      ├─KlecSpeech_train_D12_wav_0.zip | 3 GB | 48413
            │      ├─KlecSpeech_train_D13_wav_0.zip | 3 GB | 48414
            │      ├─KlecSpeech_train_D11_wav_0.zip | 3 GB | 48412
            │      ├─KlecSpeech_train_D10_wav_0.zip | 4 GB | 48411
            │      ├─KlecSpeech_train_D09_wav_0.zip | 4 GB | 48410
            │      ├─KlecSpeech_train_D08_wav_0.zip | 5 GB | 48409
            │      ├─KlecSpeech_train_D07_wav_0.zip | 4 GB | 48408
            │      ├─KlecSpeech_train_D06_wav_0.zip | 5 GB | 48407
            │      └─KlecSpeech_train_D05_wav_3.zip | 7 GB | 48406
            └─2.Validation
                ├─라벨링데이터_0908_add
                │  ├─KlecSpeech_valid_D19_label_0.zip | 163 KB | 48349
                │  ├─KlecSpeech_valid_D99_label_0.zip | 97 KB | 48350
                │  ├─KlecSpeech_valid_D01_label_0.zip | 1 MB | 48331
                │  ├─KlecSpeech_valid_D02_label_0.zip | 2 MB | 48332
                │  ├─KlecSpeech_valid_D03_label_0.zip | 1 MB | 48333
                │  ├─KlecSpeech_valid_D04_label_0.zip | 1 MB | 48334
                │  ├─KlecSpeech_valid_D05_label_0.zip | 2 MB | 48335
                │  ├─KlecSpeech_valid_D06_label_0.zip | 57 KB | 48336
                │  ├─KlecSpeech_valid_D07_label_0.zip | 76 KB | 48337
                │  ├─KlecSpeech_valid_D08_label_0.zip | 79 KB | 48338
                │  ├─KlecSpeech_valid_D09_label_0.zip | 71 KB | 48339
                │  ├─KlecSpeech_valid_D10_label_0.zip | 79 KB | 48340
                │  ├─KlecSpeech_valid_D11_label_0.zip | 58 KB | 48341
                │  ├─KlecSpeech_valid_D12_label_0.zip | 100 KB | 48342
                │  ├─KlecSpeech_valid_D13_label_0.zip | 114 KB | 48343
                │  ├─KlecSpeech_valid_D14_label_0.zip | 508 KB | 48344
                │  ├─KlecSpeech_valid_D15_label_0.zip | 147 KB | 48345
                │  ├─KlecSpeech_valid_D16_label_0.zip | 164 KB | 48346
                │  ├─KlecSpeech_valid_D17_label_0.zip | 88 KB | 48347
                │  └─KlecSpeech_valid_D18_label_0.zip | 164 KB | 48348
                └─원천데이터_0908_add
                    ├─KlecSpeech_valid_D01_wav_0.zip | 815 MB | 48351
                    ├─KlecSpeech_valid_D02_wav_0.zip | 986 MB | 48352
                    ├─KlecSpeech_valid_D03_wav_0.zip | 744 MB | 48353
                    ├─KlecSpeech_valid_D04_wav_0.zip | 611 MB | 48354
                    ├─KlecSpeech_valid_D05_wav_0.zip | 872 MB | 48355
                    ├─KlecSpeech_valid_D06_wav_0.zip | 40 MB | 48356
                    ├─KlecSpeech_valid_D07_wav_0.zip | 43 MB | 48357
                    ├─KlecSpeech_valid_D08_wav_0.zip | 42 MB | 48358
                    ├─KlecSpeech_valid_D09_wav_0.zip | 36 MB | 48359
                    ├─KlecSpeech_valid_D10_wav_0.zip | 39 MB | 48360
                    ├─KlecSpeech_valid_D11_wav_0.zip | 32 MB | 48361
                    ├─KlecSpeech_valid_D12_wav_0.zip | 62 MB | 48362
                    ├─KlecSpeech_valid_D13_wav_0.zip | 53 MB | 48363
                    ├─KlecSpeech_valid_D14_wav_0.zip | 201 MB | 48364
                    ├─KlecSpeech_valid_D15_wav_0.zip | 65 MB | 48365
                    ├─KlecSpeech_valid_D16_wav_0.zip | 59 MB | 48366
                    ├─KlecSpeech_valid_D17_wav_0.zip | 33 MB | 48367
                    ├─KlecSpeech_valid_D18_wav_0.zip | 67 MB | 48368
                    ├─KlecSpeech_valid_D19_wav_0.zip | 52 MB | 48369
                    └─KlecSpeech_valid_D99_wav_0.zip | 63 MB | 48370
"""

ds_logging.set_verbosity_info()
logger = ds_logging.get_logger("datasets")


class KlecSpeech(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="ASR",
            data_dir="ASR",
            version="1.1.0",
            description="""한국어로 된 강의영상/음성을 인식하여 자동으로 자막을 생성해주고, 내용을 이해하는 강의 음성 데이터
한국인의 음성을 문자로 바꾸어 주고, 문맥을 이해하는 한국어 음성 언어처리 기술 개발을 위한 AI 학습용 한국어 음성 DB를 구축
한국어로 된 강의 영상/음성을 인식하여 자동으로 자막을 생성해주고, 내용을 이해하는 서비스를 위한 한국어 강의 음성 DB를 구축"""
            + DESCRIPTION,
        )
    ]
    DEFAULT_CONFIG_NAME = "ASR"
    DEFAULT_WRITER_BATCH_SIZE = 1000

    def _info(self) -> DatasetInfo:
        if self.config.name == "ASR":
            features = Features(
                {
                    "id": Value("string"),
                    "audio": Audio(SAMPLE_RATE),
                    "sentence": Value("string"),
                }
            )
        return DatasetInfo(
            description=self.config.description,
            version=self.config.version,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    def aihub_downloader(self, download_path: Path) -> List[Path]:
        def download_from_aihub(download_path: Path, apikey: str) -> None:
            try:
                with TarFile.open(download_path, "r") as tar:
                    tar.getmembers()
                    return None
            except Exception as e:
                msg = f"tar 파일이 손상되었다. {e} 손상된 파일은 삭제하고 다시 다운로드 받는다."
                logger.warning(msg)
                download_path.unlink()

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
            for part_path in Path(data_dir).rglob("*.zip.part*"):
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

    def _split_generators(self, dl_manager) -> List[SplitGenerator]:  # type: ignore
        cache_dir = Path(dl_manager.download_config.cache_dir)
        download_path = cache_dir.joinpath(f"{_DATANAME}.tar")
        src_path_ls = self.aihub_downloader(download_path)

        if self.config.name == "ASR":
            train_src_ls = [path for path in src_path_ls if "1.Training" in path.as_posix()]
            valid_src_ls = [path for path in src_path_ls if "2.Validation" in path.as_posix()]
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
        if self.config.name == "ASR":
            for idx, data in enumerate(self._asr_generate_examples(**kwagrs)):
                yield idx, data

    def _asr_generate_examples(self, file_ls: List[Path], split: str) -> Generator:
        lbl_zip_ls = natsorted(
            [ZipFile(path) for path in file_ls if "라벨링데이터" in path.as_posix()],
            key=lambda x: Path(x.filename).stem,
        )
        src_zip_ls = natsorted(
            [ZipFile(path) for path in file_ls if "원천데이터" in path.as_posix()],
            key=lambda x: Path(x.filename).stem,
        )

        for label_zip, source_zip in zip(lbl_zip_ls, src_zip_ls):
            label_info_ls = natsorted(
                [info for info in label_zip.infolist() if not info.is_dir() and "json" not in info.filename],
                key=lambda x: x.filename,
            )
            source_info_ls = natsorted(
                [info for info in source_zip.infolist() if not info.is_dir()],
                key=lambda x: x.filename,
            )

            for label_info, source_info in zip(label_info_ls, source_info_ls):
                id_src, id_lbl = Path(source_info.filename), Path(label_info.filename)
                if id_src.as_posix().replace(id_src.suffix, "") != id_lbl.as_posix().replace(id_lbl.suffix, ""):
                    msg = f"라벨링 데이터와 오디오 데이터가 매칭되지 않습니다. {label_info.filename}, {source_info.filename}"
                    logger.warning(msg)
                    continue
                data_id = id_src.as_posix().replace(id_src.suffix, "")
                with label_zip.open(label_info) as lbl_f, source_zip.open(source_info) as src_f:
                    yield {
                        "id": data_id,
                        "audio": src_f.read(),
                        "sentence": lbl_f.read().decode("utf-8"),
                    }
