import io
import json
import os
from decimal import Decimal
from pathlib import Path
from tarfile import TarFile
from typing import Generator, List
from zipfile import ZipFile

import requests
import soundfile as sf
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


_LICENSE = """AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다."""


DATASET_KEY = "464"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/{DATASET_KEY}.do"
_HOMEPAGE = f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"  #  no lint


_DATANAME = "MeetingSpeech"
DATASET_SIZE = 563.73  # GB
SAMPLE_RATE = 16000


_DESCRIPTION = """한국어로 된 회의 음성을 인식하여 자동으로 회의록을 작성하고 자막을 생성하여 회의 내용 이해 서비스 제공을 위한 한국어 회의 음성 DB 구축을 목표로 다양한 실제 회의 환경과 방송, UCC의 영상 및 음원 데이터를 활용한 데이터셋 구축"""

ds_logging.set_verbosity_info()
logger = ds_logging.get_logger("datasets")


# https://github.com/huggingface/datasets/blob/dcd01046388fc052d37acc5a450bea69e3c57afc/templates/new_dataset_script.py#L65 참고해서 만듬.
class MeetingSpeech(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="ASR",
            version="1.4.0",
            description="STT 학습에 맞춰서 최적화된 데이터" + _DESCRIPTION,
        ),
    ]

    DEFAULT_CONFIG_NAME = "ASR"
    DEFAULT_WRITER_BATCH_SIZE = 1000

    def _info(self) -> DatasetInfo:
        features = Features(
            {
                "audio": Audio(SAMPLE_RATE),
                "id": Value("string"),
                "sentence": Value("string"),
                "original_form": Value("string"),
                "start": Value("float32"),
                "end": Value("float32"),
                "term": Value("string"),
                "environment": Value("string"),
                "isIdiom": Value("bool"),
                "hangeulToEnglish": [
                    {
                        "id": Value("int16"),
                        "hangeul": Value("string"),
                        "english": Value("string"),
                        "begin": Value("int16"),
                        "end": Value("int16"),
                    }
                ],
                "hangeulToNumber": [
                    {
                        "id": Value("int16"),
                        "hangeul": Value("string"),
                        "number": Value("string"),
                        "begin": Value("int16"),
                        "end": Value("int16"),
                    }
                ],
                "speaker": {
                    "id": Value("string"),
                    "name": Value("string"),
                    "age": Value("string"),
                    "occupation": Value("string"),
                    "role": Value("string"),
                    "sex": Value("string"),
                },
                "metadata": {
                    "title": Value("string"),
                    "creator": Value("string"),
                    "distributor": Value("string"),
                    "year": Value("int16"),
                    "category": Value("string"),
                    "sampling": Value("string"),
                    "date": Value("string"),
                    "topic": Value("string"),
                    "media": Value("string"),
                    "communication": Value("string"),
                    "type": Value("string"),
                    "domain": Value("string"),
                    "speaker_num": Value("int16"),
                    "organization": Value("string"),
                    "annotation_level": Value("string"),
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

    def _split_generators(self, dl_manager) -> List[SplitGenerator]:
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
        source_ls = [ZipFile(x) for x in file_ls if "원천데이터" in str(x)]
        label_ls = [ZipFile(x) for x in file_ls if "라벨링데이터" in str(x)]

        source_ls = natsorted(source_ls, key=lambda x: x.filename)
        label_ls = natsorted(label_ls, key=lambda x: x.filename)

        info_replacer = lambda info, key: info.filename.split("/")[-1].replace(key, "")  # noqa

        for audio_zip, label_zip in zip(source_ls, label_ls):
            audio_filelist = [x for x in audio_zip.filelist if "wav" in x.filename]
            label_filelist = [x for x in label_zip.filelist if not x.is_dir()]

            audio_filelist = natsorted(audio_filelist, key=lambda x: x.filename)
            label_filelist = natsorted(label_filelist, key=lambda x: x.filename)
            label_dict = {info_replacer(x, ".json"): x for x in label_filelist}

            for audio_info in audio_filelist:
                audio_file_name = info_replacer(audio_info, ".wav")
                if audio_file_name not in label_dict:
                    print(f"{audio_file_name}와 매칭되는 파일이 없음. 해당 파일은 스킵함.")
                    continue
                label_info = label_dict[audio_file_name]

                raw_label_data = label_zip.open(label_info).read()
                raw_audio_data = audio_zip.open(audio_info).read()

                label = json.loads(raw_label_data.decode("utf-8"))
                try:
                    audio, sr = sf.read(io.BytesIO(raw_audio_data))
                    audio_info = sf.info(io.BytesIO(raw_audio_data))
                except sf.LibsndfileError:
                    print(f"{audio_file_name} is corrupted! processing was passed!!")
                    continue

                speakers_dict = {x["id"]: x for x in label["speaker"]}
                metadata = label["metadata"]

                for speech_part in label["utterance"]:
                    speech_part: dict  # for intellisense
                    speaker_id = speech_part.pop("speaker_id")
                    form = speech_part.pop("form")

                    speech_part["speaker"] = speakers_dict[speaker_id] if speaker_id in speakers_dict else None
                    speech_part["metadata"] = metadata

                    # ETRI 전사규칙에 따라 철자는 오른쪽, 발음은 왼쪽으로 바꿔야 함.
                    if speech_part["hangeulToNumber"]:
                        hangeulToNumber = list({x["hangeul"]: x for x in speech_part["hangeulToNumber"]}.values())

                        for example in hangeulToNumber:
                            find_regex = f"""({example["hangeul"]})/({example["number"]})"""
                            find_regex_1 = f"""(@{example["hangeul"]})/(@{example["number"]})"""
                            find_regex_2 = f"""({example["hangeul"]})/(@{example["number"]})"""
                            find_regex_3 = f"""(@{example["hangeul"]})/({example["number"]})"""

                            replace_regex = f"""({example["number"]})/({example["hangeul"]})"""

                            if replace_regex in form:
                                # 내가 원하는 이중전사가 포함된 경우 스킵
                                continue

                            if (
                                (find_regex not in form)
                                and (find_regex_1 not in form)
                                and (find_regex_2 not in form)
                                and (find_regex_3 not in form)
                            ):
                                error_msg = (
                                    f"이중 전사가 잘못된 녀석: {form}\n"
                                    f"regex_1: {find_regex}\n"
                                    f"regex_2: {find_regex_1}"
                                    f"regex_3: {find_regex_2}"
                                    f"regex_4: {find_regex_3}"
                                    f"replace_regex: {replace_regex}"
                                    f"""hangeulToNumber: {speech_part["hangeulToNumber"]}"""
                                )
                                print(error_msg)
                                continue
                                raise ValueError(error_msg)

                            form = form.replace(find_regex, replace_regex)
                            form = form.replace(find_regex_1, replace_regex)
                            form = form.replace(find_regex_2, replace_regex)
                            form = form.replace(find_regex_3, replace_regex)

                    if speech_part["hangeulToEnglish"]:
                        hangeulToEnglish = list({x["hangeul"]: x for x in speech_part["hangeulToEnglish"]}.values())
                        for example in hangeulToEnglish:
                            find_regex = f"""({example["hangeul"]})/({example["english"]})"""
                            find_regex_1 = f"""(@{example["hangeul"]})/(@{example["english"]})"""
                            find_regex_2 = f"""({example["hangeul"]})/(@{example["english"]})"""
                            find_regex_3 = f"""(@{example["hangeul"]})/({example["english"]})"""

                            replace_regex = f"""({example["english"]})/({example["hangeul"]})"""
                            if replace_regex in form:
                                # 내가 원하는 이중전사가 포함된 경우 스킵
                                continue
                            if (
                                (find_regex not in form)
                                and (find_regex_1 not in form)
                                and (find_regex_2 not in form)
                                and (find_regex_3 not in form)
                            ):
                                error_msg = (
                                    f"이중 전사가 잘못된 녀석: {form}\n"
                                    f"regex_1: {find_regex}\n"
                                    f"regex_2: {find_regex_1}"
                                    f"regex_3: {find_regex_2}"
                                    f"regex_4: {find_regex_3}"
                                    f"replace_regex: {replace_regex}"
                                    f"""hangeulToEnglish: {speech_part["hangeulToEnglish"]}"""
                                )
                                print(error_msg)
                                continue
                                raise ValueError(error_msg)

                            form = form.replace(find_regex, replace_regex)
                            form = form.replace(find_regex_1, replace_regex)
                            form = form.replace(find_regex_2, replace_regex)
                            form = form.replace(find_regex_3, replace_regex)
                    speech_part["sentence"] = form

                    speech_start = round(Decimal(speech_part["start"]) * sr)
                    speech_end = round(Decimal(speech_part["end"]) * sr)

                    speech_part["start"] = float(speech_part["start"])
                    speech_part["end"] = float(speech_part["end"])

                    speech_array = audio[speech_start:speech_end]

                    speech_byte = io.BytesIO()
                    sf.write(
                        file=speech_byte,
                        data=speech_array,
                        samplerate=sr,
                        subtype=audio_info.subtype,
                        endian=audio_info.endian,
                        format=audio_info.format,
                    )

                    speech_part["audio"] = speech_byte.getvalue()

                    yield speech_part
