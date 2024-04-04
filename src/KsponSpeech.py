# -*- coding: utf-8 -*-
import io
import os
import re
import wave
from pathlib import Path
from tarfile import TarFile
from typing import List
from zipfile import ZipFile

import numpy as np
import requests
import soundfile as sf
from datasets import (
    Audio,
    DatasetInfo,
    Features,
    GeneratorBasedBuilder,
    NamedSplit,
    SplitGenerator,
    Value,
)
from natsort import natsorted
from tqdm import tqdm

_LICENSE = """
AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다.
"""

_CITATION = """\
@article{bang2020ksponspeech,
  title={KsponSpeech: Korean spontaneous speech corpus for automatic speech recognition},
  author={Bang, Jeong-Uk and Yun, Seung and Kim, Seung-Hi and Choi, Mu-Yeol and Lee, Min-Kyu and Kim, Yeo-Jeong and Kim, Dong-Hyun and Park, Jun and Lee, Young-Jik and Kim, Sang-Hun},
  journal={Applied Sciences},
  volume={10},
  number={19},
  pages={6936},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
}
"""

_DESCRIPTION = """
KsponSpeech is a large-scale spontaneous speech corpus of Korean conversations. This corpus contains 969 hrs of general open-domain dialog utterances, spoken by about 2,000 native Korean speakers in a clean environment. All data were constructed by recording the dialogue of two people freely conversing on a variety of topics and manually transcribing the utterances. The transcription provides a dual transcription consisting of orthography and pronunciation, and disfluency tags for spontaneity of speech, such as filler words, repeated words, and word fragments. KsponSpeech is publicly available on an open data hub site of the Korea government. (https://aihub.or.kr/aidata/105)
"""


DATASET_KEY = "123"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/{DATASET_KEY}.do"
_HOMEPAGE = (
    f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"
)

search_audio_zip = re.compile("KsponSpeech_(0[1-5]|eval).zip")

_VERSION = "1.0.0"
_DATANAME = "KsponSpeech"


class KsponSpeech(GeneratorBasedBuilder):
    def _info(self) -> DatasetInfo:
        features = Features(
            {
                "audio": Audio(16000),
                "sentence": Value("string"),
                "id": Value("string"),
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

        # TODO: need clean code!!!
        audio_file_info = dict()
        for path in zip_file_path:
            if not search_audio_zip.findall(str(path)):
                continue

            zip_file = ZipFile(path)
            audio_file_info.update(
                {file.filename.split("/")[-1]: file for file in zip_file.filelist if "pcm" in file.filename}
            )

        label_zip = [ZipFile(str(x)) for x in zip_file_path if "KsponSpeech_scripts" in str(str(x))][0]

        get_zip_file_name: str = lambda x: x.stem.split("/")[-1].replace(".zip", "")
        audio_zip_dict = {get_zip_file_name(x): ZipFile(x) for x in zip_file_path if search_audio_zip.findall(str(x))}

        for zip_info in label_zip.filelist:
            label = label_zip.open(zip_info).read().decode("utf-8")
            # [:2]는 split("\n") 했을 때 마지막 행에 null 값이 들어가는 걸 방지하기 위한 목적
            label = label[:-2]  # 마지막 열에 있는 개행문자 제거
            label_ls = label.split("\n")

            # NOTE: 파일 이름이 [train, dev, eval_clean, clean_other]로 되어 있음
            file_name = zip_info.filename.split(".")[-2]

            yield SplitGenerator(
                name=NamedSplit(file_name),
                gen_kwargs={
                    "label_ls": label_ls,
                    "audio_file_info": audio_file_info,
                    "audio_zip_dict": audio_zip_dict,
                    "split": file_name,
                },
            )

    def _generate_examples(
        self,
        label_ls: list,
        audio_file_info: dict,
        audio_zip_dict: dict,
        split: str,
    ):
        for _id, label in enumerate(label_ls):
            path, sentence = tuple(label.split(" :: "))

            # KsponSpeech_05/KsponSpeech_0623/KsponSpeech_622536.pcm
            # 0: 카테고리, 1: 데이터 폴더, 2: 파일
            path_segment = path.split("/")
            pcm_audio = audio_zip_dict[path_segment[0]].open(audio_file_info[path_segment[-1]]).read()
            try:

                bytes_value = np.frombuffer(pcm_audio, dtype=np.int16).astype(np.float32) / 32767
                buffer = io.BytesIO(bytes())
                # ksponspeech는 무조건 고정임.
                sf.write(buffer, bytes_value, 16000, format="wav")
            except ValueError:
                buffer = io.BytesIO(bytes())

                with wave.open(buffer, "wb") as wave_file:
                    wave_file.setnchannels(1)
                    wave_file.setsampwidth(16 // 8)
                    wave_file.setframerate(16000)
                    wave_file.writeframes(pcm_audio)

            data = {"audio": buffer.getvalue(), "sentence": sentence, "id": path_segment[-1].replace(".pcm", "")}

            yield (_id, data)
