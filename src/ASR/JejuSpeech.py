# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
from tarfile import TarFile
from typing import List
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
import soundfile as sf
import io
from natsort import natsorted
from tqdm import tqdm

_LICENSE = """
AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다.
"""

_CITATION = None

_DESCRIPTION = """\
방언(제주도)을 사용하는 일상 대화를 인식, 음성을 문자로 바꾸어주는 방언 발화 음성 데이터
"""


DATASET_KEY = "121"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/{DATASET_KEY}.do"
_HOMEPAGE = f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"

_VERSION = "1.2.0"
_DATANAME = "JejuSpeech"


# https://github.com/huggingface/datasets/blob/dcd01046388fc052d37acc5a450bea69e3c57afc/templates/new_dataset_script.py#L65 참고해서 만듬.
class JejuSpeech(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="STT", version=_VERSION, description="STT 학습에 맞춰서 최적화된 데이터"
        ),
        # VAD에 사용할 수 있지 않을까 해서 이렇게 남겨 둠.
        BuilderConfig(name="Original", version=_VERSION, description="순수 raw 데이터"),
    ]

    DEFAULT_CONFIG_NAME = "STT"

    def _info(self) -> DatasetInfo:
        if self.config.name == "STT":
            features = Features(
                {
                    "id": Value("string"),
                    "audio": Audio(16000),
                    "sentence": Value("string"),
                    "standard_form": Value("string"),
                    "dialect_form": Value("string"),
                    "start": Value("float32"),
                    "end": Value("float32"),
                    "note": Value("string"),
                    "eojeolList": [
                        {
                            "id": Value("int8"),
                            "eojeol": Value("string"),
                            "standard": Value("string"),
                            "isDialect": Value("bool"),
                        }
                    ],
                    "speaker": {
                        "id": Value("string"),
                        "name": Value("string"),
                        "age": Value("string"),
                        "occupation": Value("string"),
                        "sex": Value("string"),
                        "birthplace": Value("string"),
                        "principal_residence": Value("string"),
                        "current_residence": Value("string"),
                        "education": Value("string"),
                    },
                    "metadata": {
                        "title": Value("string"),
                        "creator": Value("string"),
                        "distributor": Value("string"),
                        "year": Value("string"),
                        "category": Value("string"),
                        "annotation_level": [Value("string")],
                        "sampling": Value("string"),
                        "author": Value("string"),
                        "publisher": Value("string"),
                        "date": Value("string"),
                        "topic": Value("string"),
                    },
                }
            )
        elif self.config.name == "Original":
            features = Features(
                {
                    "audio": Audio(16000),
                    "id": Value("string"),
                    "metadata": {
                        "title": Value("string"),
                        "creator": Value("string"),
                        "distributor": Value("string"),
                        "year": Value("string"),
                        "category": Value("string"),
                        "annotation_level": [Value("string")],
                        "sampling": Value("string"),
                        "author": Value("string"),
                        "publisher": Value("string"),
                        "date": Value("string"),
                        "topic": Value("string"),
                    },
                    "speaker": [
                        {
                            "id": Value("string"),
                            "name": Value("string"),
                            "age": Value("string"),
                            "occupation": Value("string"),
                            "sex": Value("string"),
                            "birthplace": Value("string"),
                            "principal_residence": Value("string"),
                            "current_residence": Value("string"),
                            "education": Value("string"),
                        }
                    ],
                    "setting": {"relation": Value("string")},
                    "utterance": [
                        {
                            "id": Value("string"),
                            "form": Value("string"),
                            "standard_form": Value("string"),
                            "dialect_form": Value("string"),
                            "start": Value("float32"),
                            "end": Value("float32"),
                            "note": Value("string"),
                            "eojeolList": [
                                {
                                    "id": Value("int8"),
                                    "eojeol": Value("string"),
                                    "standard": Value("string"),
                                    "isDialect": Value("bool"),
                                }
                            ],
                        }
                    ],
                }
            )
        else:
            raise NotImplementedError()

        return DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            version=_VERSION,
        )

    def aihub_downloader(self, recv_path: Path) -> None:
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

        if response.status_code != 200:
            raise BaseException(f"Download failed with HTTP status code: {response.status_code}")

        with open(recv_path, "wb") as file:
            # chunk_size는 byte수
            for chunk in tqdm(response.iter_content(chunk_size=1024)):
                file.write(chunk)

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

    def _original_generate_examples(self, filepath: List[Path], split: str):
        source_ls = [ZipFile(x) for x in filepath if "원천데이터" in str(x)]
        label_ls = [ZipFile(x) for x in filepath if "라벨링데이터" in str(x)]

        source_ls = natsorted(source_ls, key=lambda x: x.filename)
        label_ls = natsorted(label_ls, key=lambda x: x.filename)

        info_replacer = lambda info, key: info.filename.replace(key, "")

        label_zip = label_ls[0]
        label_dict = {
            info_replacer(info, ".json"): info
            for info in label_zip.filelist
            if "json" in info.filename
        }

        id_counter = 0
        for audio_zip in source_ls:
            for audio_info in audio_zip.filelist:
                audio_file_name = info_replacer(audio_info, ".wav")
                # 일부 음성에 라벨 파일이 누락된 경우가 존재함. 라벨이 누락된 음성에 대해선 데이터를 생성하지 않고 pass 함.
                if audio_file_name not in label_dict:
                    continue

                label_info = label_dict[audio_file_name]

                raw_label_data = label_zip.open(label_info).read()
                raw_audio_data = audio_zip.open(audio_info).read()

                label = label = json.loads(
                    raw_label_data.decode(json.detect_encoding(raw_label_data))
                )

                label["audio"] = raw_audio_data

                yield (id_counter, label)
                id_counter += 1

    def _stt_generate_examples(self, filepath: List[Path], split: str):
        source_ls = [ZipFile(x) for x in filepath if "원천데이터" in str(x)]
        label_ls = [ZipFile(x) for x in filepath if "라벨링데이터" in str(x)]

        source_ls = natsorted(source_ls, key=lambda x: x.filename)
        label_ls = natsorted(label_ls, key=lambda x: x.filename)

        info_replacer = lambda info, key: info.filename.replace(key, "")

        label_zip = label_ls[0]
        label_dict = {
            info_replacer(info, ".json"): info
            for info in label_zip.filelist
            if "json" in info.filename
        }

        id_counter = 0
        for audio_zip in source_ls:
            for audio_info in audio_zip.filelist:
                audio_file_name = info_replacer(audio_info, ".wav")
                # 일부 음성에 라벨 파일이 누락된 경우가 존재함. 라벨이 누락된 음성에 대해선 데이터를 생성하지 않고 pass 함.
                if audio_file_name not in label_dict:
                    print(f"{audio_file_name} not have label file, processing was passed!!")
                    continue

                label_info = label_dict[audio_file_name]

                raw_label_data = label_zip.open(label_info).read()
                raw_audio_data = audio_zip.open(audio_info).read()

                label = json.loads(raw_label_data.decode(json.detect_encoding(raw_label_data)))
                try:
                    audio, sr = sf.read(io.BytesIO(raw_audio_data))
                except sf.LibsndfileError:
                    print(f"{audio_file_name} is corrupted! processing was passed!!")
                    continue
                audio_info = sf.info(io.BytesIO(raw_audio_data))

                speakers_dict = {x["id"]: x for x in label["speaker"]}

                for speech_part in label["utterance"]:
                    speech_part: dict  # for intellisense
                    speaker_id = speech_part.pop("speaker_id")
                    form = speech_part.pop("form")
                    if speaker_id and (speaker_id in speakers_dict):
                        speech_part["speaker"] = speakers_dict[speaker_id]
                    else:
                        speech_part["speaker"] = None
                    speech_part["metadata"] = label["metadata"]
                    speech_part["sentence"] = form

                    # OPTIMIZE: round 때문에 음성이 겹칠 수 있는 위험이 있다.
                    speech_start = round(speech_part["start"] * sr)
                    speech_end = round(speech_part["end"] * sr)

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

    def _generate_examples(self, **kwargs):
        if self.config.name == "STT":
            return self._stt_generate_examples(**kwargs)
        elif self.config.name == "Original":
            return self._original_generate_examples(**kwargs)
        else:
            raise NotImplementedError()
