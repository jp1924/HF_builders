# -*- coding: utf-8 -*-
import io
import json
import os
from decimal import Decimal
from pathlib import Path
from tarfile import TarFile
from typing import List
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
from natsort import natsorted
from tqdm import tqdm

_LICENSE = """
AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다.
"""

_CITATION = None

_DESCRIPTION = """\
한국어로 된 회의 음성을 인식하여 자동으로 회의록을 작성하고 자막을 생성하여 회의 내용 이해 서비스 제공을 위한 한국어 회의 음성 DB 구축을 목표로 다양한 실제 회의 환경과 방송, UCC의 영상 및 음원 데이터를 활용한 데이터셋 구축
"""


DATASET_KEY = "464"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/{DATASET_KEY}.do"
_HOMEPAGE = f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"

_VERSION = "1.4.0"
_DATANAME = "MeetingSpeech"
DATASET_SIZE = 563.73


# https://github.com/huggingface/datasets/blob/dcd01046388fc052d37acc5a450bea69e3c57afc/templates/new_dataset_script.py#L65 참고해서 만듬.
class MeetingSpeech(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="STT", version=_VERSION, description="STT 학습에 맞춰서 최적화된 데이터"
        ),
    ]

    DEFAULT_CONFIG_NAME = "STT"

    def _info(self) -> DatasetInfo:
        features = Features(
            {
                "audio": Audio(16000),
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
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            version=_VERSION,
        )

    def aihub_downloader(self, destination_path: Path) -> None:
        aihub_id = os.getenv("AIHUB_ID", None)
        aihub_pass = os.getenv("AIHUB_PASS", None)

        if not aihub_id:
            raise ValueError(
                """AIHUB_ID가 지정되지 않았습니다. `os.environ["AIHUB_ID"]="your_id"`로 ID를 지정해 주세요"""
            )
        if not aihub_pass:
            raise ValueError(
                """AIHUB_PASS가 지정되지 않았습니다. `os.environ["AIHUB_PASS"]="your_pass"`로 ID를 지정해 주세요"""
            )

        response = requests.get(
            DOWNLOAD_URL,
            headers={"id": aihub_id, "pass": aihub_pass},
            params={"fileSn": "all"},
            stream=True,
        )

        if response.status_code == 502:
            raise BaseException(
                "다운로드 서비스는 홈페이지(https://aihub.or.kr)에서 신청 및 승인 후 이용 가능 합니다."
            )
        elif response.status_code != 200:
            raise BaseException(f"Download failed with HTTP status code: {response.status_code}")

        data_file = open(destination_path, "wb")
        downloaded_bytes = 0
        with tqdm(total=round(DATASET_SIZE * 1024**2)) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                data_file.write(chunk)
                downloaded_bytes += len(chunk)

                pbar.update(1)
                prefix = f"Downloaded (GB): {downloaded_bytes / (1024**3):.4f}/{DATASET_SIZE}"
                pbar.set_postfix_str(prefix)

        data_file.close()

    def concat_zip_part(self, unzip_dir: Path) -> None:
        part_glob = Path(unzip_dir).rglob("*.zip.part*")

        part_dict = dict()
        for part_path in part_glob:
            parh_stem = str(part_path.parent.joinpath(part_path.stem))

            if parh_stem not in part_dict:
                part_dict[parh_stem] = list()

            part_dict[parh_stem].append(part_path)

        for dst_path, part_path_ls in part_dict.items():
            with open(dst_path, "wb") as byte_f:
                for part_path in natsorted(part_path_ls):
                    byte_f.write(part_path.read_bytes())
                    os.remove(part_path)

    def _split_generators(self, dl_manager) -> List[SplitGenerator]:  # type: ignore
        cache_dir = Path(dl_manager.download_config.cache_dir)

        unzip_dir = cache_dir.joinpath(_DATANAME)
        tar_file = cache_dir.joinpath(f"{_DATANAME}.tar")

        if tar_file.exists():
            os.remove(tar_file)

        if not unzip_dir.exists():
            self.aihub_downloader(tar_file)

            with TarFile(tar_file, "r") as mytar:
                mytar.extractall(unzip_dir)
                os.remove(tar_file)

            self.concat_zip_part(unzip_dir)

        zip_file_path = list(unzip_dir.rglob("*.zip"))

        train_split = [x for x in zip_file_path if "Training" in str(x)]
        valid_split = [x for x in zip_file_path if "Validation" in str(x)]

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "filepath": train_split,
                    "split": "train",
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "filepath": valid_split,
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(self, filepath: List[Path], split: str):
        source_ls = [ZipFile(x) for x in filepath if "원천데이터" in str(x)]
        label_ls = [ZipFile(x) for x in filepath if "라벨링데이터" in str(x)]

        source_ls = natsorted(source_ls, key=lambda x: x.filename)
        label_ls = natsorted(label_ls, key=lambda x: x.filename)

        info_replacer = lambda info, key: info.filename.split("/")[-1].replace(key, "")

        id_counter = 0
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

                    speech_part["speaker"] = (
                        speakers_dict[speaker_id] if speaker_id in speakers_dict else None
                    )
                    speech_part["metadata"] = metadata

                    # ETRI 전사규칙에 따라 철자는 오른쪽, 발음은 왼쪽으로 바꿔야 함.
                    if speech_part["hangeulToNumber"]:
                        hangeulToNumber = list(
                            {x["hangeul"]: x for x in speech_part["hangeulToNumber"]}.values()
                        )

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
                        hangeulToEnglish = list(
                            {x["hangeul"]: x for x in speech_part["hangeulToEnglish"]}.values()
                        )
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

                    yield (id_counter, speech_part)
                    id_counter += 1
