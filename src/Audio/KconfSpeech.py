import json
import os
from copy import deepcopy
from pathlib import Path
from tarfile import TarFile
from typing import Generator, List
from zipfile import ZipFile

import requests
from datasets import Audio, BuilderConfig, DatasetInfo, Features, GeneratorBasedBuilder, Split, SplitGenerator, Value
from datasets import logging as ds_logging
from natsort import natsorted
from tqdm import tqdm


_LICENSE = """AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다."""


DATASET_KEY = "132"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/0.5/{DATASET_KEY}.do"
_HOMEPAGE = f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"  # nolint


_DATANAME = "KconfSpeech"
DATASET_SIZE = 248.33  # GB
SAMPLE_RATE = 16000


DESCRIPTION = """└─010.회의 음성 데이터
        └─01.데이터
            ├─1.Training
            │  ├─라벨링데이터_0908_add
            │  │  ├─KconfSpeech_train_D20_label_1.zip | 19 MB | 48304
            │  │  ├─KconfSpeech_train_D21_label_0.zip | 48 MB | 48305
            │  │  ├─KconfSpeech_train_D21_label_1.zip | 16 MB | 48306
            │  │  ├─KconfSpeech_train_D22_label_0.zip | 47 MB | 48307
            │  │  ├─KconfSpeech_train_D22_label_1.zip | 19 MB | 48308
            │  │  ├─KconfSpeech_train_D23_label_0.zip | 50 MB | 48309
            │  │  ├─KconfSpeech_train_D23_label_1.zip | 42 MB | 48310
            │  │  ├─KconfSpeech_train_D24_label_0.zip | 37 MB | 48311
            │  │  ├─KconfSpeech_train_D25_label_0.zip | 42 MB | 48312
            │  │  ├─KconfSpeech_train_D25_label_1.zip | 44 MB | 48313
            │  │  ├─KconfSpeech_train_D25_label_2.zip | 4 MB | 48314
            │  │  ├─KconfSpeech_train_D26_label_0.zip | 36 MB | 48315
            │  │  ├─KconfSpeech_train_D27_label_0.zip | 47 MB | 48316
            │  │  ├─KconfSpeech_train_D27_label_1.zip | 4 MB | 48317
            │  │  └─KconfSpeech_train_D20_label_0.zip | 47 MB | 48303
            │  └─원천데이터_0908_add
            │      ├─KconfSpeech_train_D20_wav_0.zip | 24 GB | 48318
            │      ├─KconfSpeech_train_D20_wav_1.zip | 9 GB | 48319
            │      ├─KconfSpeech_train_D21_wav_0.zip | 23 GB | 48320
            │      ├─KconfSpeech_train_D21_wav_1.zip | 8 GB | 48321
            │      ├─KconfSpeech_train_D22_wav_0.zip | 23 GB | 48322
            │      ├─KconfSpeech_train_D22_wav_1.zip | 9 GB | 48323
            │      ├─KconfSpeech_train_D23_wav_0.zip | 23 GB | 48324
            │      ├─KconfSpeech_train_D23_wav_1.zip | 19 GB | 48325
            │      ├─KconfSpeech_train_D27_wav_1.zip | 2 GB | 48258
            │      ├─KconfSpeech_train_D24_wav_0.zip | 19 GB | 48301
            │      ├─KconfSpeech_train_D25_wav_0.zip | 22 GB | 48302
            │      ├─KconfSpeech_train_D25_wav_1.zip | 22 GB | 48254
            │      ├─KconfSpeech_train_D25_wav_2.zip | 2 GB | 48255
            │      ├─KconfSpeech_train_D26_wav_0.zip | 17 GB | 48256
            │      └─KconfSpeech_train_D27_wav_0.zip | 23 GB | 48257
            └─2.Validation
                ├─라벨링데이터_0908_add
                │  ├─KconfSpeech_valid_D20_label_0.zip | 572 KB | 48259
                │  ├─KconfSpeech_valid_D21_label_0.zip | 604 KB | 48260
                │  ├─KconfSpeech_valid_D22_label_0.zip | 696 KB | 48261
                │  ├─KconfSpeech_valid_D23_label_0.zip | 1 MB | 48262
                │  ├─KconfSpeech_valid_D24_label_0.zip | 346 KB | 48263
                │  ├─KconfSpeech_valid_D25_label_0.zip | 477 KB | 48264
                │  ├─KconfSpeech_valid_D26_label_0.zip | 340 KB | 48265
                │  └─KconfSpeech_valid_D27_label_0.zip | 485 KB | 48266
                └─원천데이터_0908_add
                    ├─KconfSpeech_valid_D20_wav_0.zip | 280 MB | 48267
                    ├─KconfSpeech_valid_D21_wav_0.zip | 305 MB | 48268
                    ├─KconfSpeech_valid_D22_wav_0.zip | 353 MB | 48269
                    ├─KconfSpeech_valid_D23_wav_0.zip | 611 MB | 48270
                    ├─KconfSpeech_valid_D24_wav_0.zip | 165 MB | 48271
                    ├─KconfSpeech_valid_D25_wav_0.zip | 240 MB | 48272
                    ├─KconfSpeech_valid_D26_wav_0.zip | 161 MB | 48273
                    └─KconfSpeech_valid_D27_wav_0.zip | 236 MB | 48274
### 파일 구조
#### 라벨링데이터
D26
 ┗ G02
 ┃ ┣ S000009
 ┃ ┃ ┣ 000000.txt
 ┃ ┃ ┗ S000009.json
 ┃ ┣ S000010
 ┃ ┃ ┣ 000000.txt
 ┃ ┃ ┣ ...
 ┃ ┃ ┗ S000010.json
 ┃ ┣ S000011
 ┃ ┃ ┣ 000000.txt
 ┃ ┃ ┣ ...
 ┃ ┃ ┗ S000011.json
 ┃ ┗ S000012
 ┃ ┃ ┣ 000000.txt
 ┃ ┃ ┣ ...
 ┃ ┃ ┗ S000012.json
## 원천데이터
D26
 ┗ G02
 ┃ ┣ S000009
 ┃ ┃ ┣ 000000.wav
 ┃ ┃ ┣ ...
 ┃ ┃ ┗ 000141.wav
 ┃ ┣ S000010
 ┃ ┃ ┣ 000000.wav
 ┃ ┃ ┣ ...
 ┃ ┃ ┗ 000106.wav
 ┃ ┣ S000011
 ┃ ┃ ┣ 000000.wav
 ┃ ┃ ┣ ...
 ┃ ┃ ┗ 000300.wav
 ┃ ┗ S000012
 ┃ ┃ ┣ 000000.wav
 ┃ ┃ ┣ ...
 ┃ ┃ ┗ 000124.wav
### 메타 데이터 구조
{
	"dataSet": {
		"version": "1.0",
		"date": "19900102",
		"typeInfo": {
			"category": "IT",
			"subcategory": "일반",
			"place": "studio",
			"speakers": [
				{
					"id": "1",
					"age": "20(?)",
					"gender": "여",
					"residence": null
				},
				{
					"id": "2",
					"age": "20(?)",
					"gender": "여",
					"residence": null
				}
			],
			"inputType": "broadcast"
		},
		"dialogs": [
			{
				"speaker": "1",
				"audioPath": "KconfSpeech/D26/G02/S000012/000000.wav",
				"textPath": "KconfSpeech/D26/G02/S000012/000000.txt"
			}
		]
	}
}"""

ds_logging.set_verbosity_info()
logger = ds_logging.get_logger("datasets")


class KconfSpeech(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="ASR",
            data_dir="ASR",
            version="1.1.0",
            description="""한국어로 된 회의영상/음성을 인식하여 자동으로 자막을 생성해주고, 내용을 이해하는 한국어 회의 음성 데이터"""
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
                    "metadata": {
                        "version": Value("string"),
                        "date": Value("string"),
                        "typeInfo": {
                            "category": Value("string"),
                            "subcategory": Value("string"),
                            "place": Value("string"),
                            "speakers": [
                                {
                                    "id": Value("string"),
                                    "gender": Value("string"),
                                    "age": Value("string"),
                                    "residence": Value("string"),
                                }
                            ],
                            "inputType": Value("string"),
                        },
                        "dialogs": [
                            {
                                "speaker": Value("string"),
                                "audioPath": Value("string"),
                                "textPath": Value("string"),
                            }
                        ],
                    },
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

    def _split_generators(self, dl_manager) -> List[SplitGenerator]:  # type: ignore
        cache_dir = Path(dl_manager.download_config.cache_dir)

        download_path = cache_dir.joinpath(f"{_DATANAME}.tar")
        src_path_ls = self._aihub_downloader(download_path)

        if self.config.name == "ASR":
            train_src_ls = [path for path in src_path_ls if "Training" in path.as_posix()]
            valid_src_ls = [path for path in src_path_ls if "Validation" in path.as_posix()]

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

    def _asr_generate_examples(self, file_ls: List[Path], split: str):
        src_zip_ls = [ZipFile(path) for path in file_ls if "원천데이터" in path.as_posix()]
        lbl_zip_ls = [ZipFile(path) for path in file_ls if "라벨링데이터" in path.as_posix()]

        src_zip_ls = natsorted(src_zip_ls, key=lambda x: x.filename)
        lbl_zip_ls = natsorted(lbl_zip_ls, key=lambda x: x.filename)

        for src_zip, lbl_zip in zip(src_zip_ls, lbl_zip_ls):
            lbl_info_ls = [info for info in lbl_zip.infolist() if not info.is_dir()]
            src_info_ls = [info for info in src_zip.infolist() if not info.is_dir()]

            lbl_dict = dict()
            for lbl_info in lbl_info_ls:
                _id = lbl_info.filename.split("/")[-2]
                lbl_dict.setdefault(_id, list()).append(lbl_info)

            src_dict = dict()
            for src_info in src_info_ls:
                _id = src_info.filename.split("/")[-2]
                src_dict.setdefault(_id, list()).append(src_info)

            for _id, wav_ls in src_dict.items():
                meta = [lbl for lbl in lbl_dict[_id] if "json" in lbl.filename][0]
                txt_ls = natsorted([lbl for lbl in lbl_dict[_id] if "txt" in lbl.filename], key=lambda x: x.filename)
                wav_ls = natsorted(wav_ls, key=lambda x: x.filename)

                meta = json.loads(lbl_zip.open(meta).read().decode("utf-8"))

                speakers = meta["dataSet"]["typeInfo"]["speakers"]
                speakers = {x["id"]: x for x in speakers}
                meta["dataSet"]["typeInfo"]["speakers"] = []

                dialogs = meta["dataSet"]["dialogs"]
                dialogs = {x["audioPath"].split("/")[-1]: x for x in dialogs}
                meta["dataSet"]["dialogs"] = []

                for txt_info, wav_info in zip(txt_ls, wav_ls):
                    txt_info_name, wav_info_name = Path(txt_info.filename), Path(wav_info.filename)

                    if txt_info_name.stem != wav_info_name.stem:
                        logger.warning(f"txt 데이터: {txt_info_name.stem}, wav 데이터: {wav_info_name.stem}")
                        continue

                    filename = wav_info.filename.split("/")[-1]
                    raw_meta = deepcopy(meta)

                    raw_meta["dataSet"]["dialogs"] = [dialogs[filename]]
                    raw_meta["dataSet"]["typeInfo"]["speakers"] = [speakers[dialogs[filename]["speaker"]]]

                    yield {
                        "id": "/".join([*txt_info_name.parts[:-1], txt_info_name.stem]),
                        "sentence": lbl_zip.open(txt_info).read().decode("utf-8"),
                        "audio": src_zip.open(wav_info).read(),
                        "metadata": raw_meta["dataSet"],
                    }
