import json
import os
from pathlib import Path
from tarfile import TarFile
from typing import Generator, List
from zipfile import ZipFile

import requests
from datasets import BuilderConfig, DatasetInfo, Features, GeneratorBasedBuilder, Split, SplitGenerator, Value
from natsort import natsorted
from tqdm import tqdm

from transformers import set_seed


set_seed(42)

_LICENSE = """
AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다.
"""

_CITATION = None

_DESCRIPTION = """\
한국어 말뭉치 데이터 20억 어절/310만 건과 Reinforcement Learning Human Feedback(RLHF) 데이터 7만 7천 건으로 구성
"""


DATASET_KEY = "71748"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/{DATASET_KEY}.do"
_HOMEPAGE = (
    f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"
)

_VERSION = "1.1.0"
_DATANAME = "DevelopmentandDataofLLMswithEnhancedKoreanLanguagePerformance"
DATASET_SIZE = 16.07


class DevelopmentandDataofLLMswithEnhancedKoreanLanguagePerformance(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(name="SFT", version=_VERSION, data_dir="SFT"),
        BuilderConfig(name="RL", version=_VERSION, data_dir="RL"),
        BuilderConfig(name="PPO", version=_VERSION, data_dir="PPO"),
    ]
    DEFAULT_CONFIG_NAME = "SFT"
    VERSION = _VERSION

    def _info(self):
        if self.config.name == "SFT":
            features = Features(
                {
                    "id": Value("string"),
                    "conversations": [{"role": Value("string"), "content": Value("string")}],
                    "prompt": Value("string"),
                    "answer": Value("string"),
                    "metadata": {
                        "main": Value("string"),
                        "middle": Value("string"),
                        "prompt_type": Value("string"),
                    },
                }
            )
        elif self.config.name == "RL":
            # HuggingFaceH4/ultrafeedback_binarized 참고
            features = Features(
                {
                    "id": Value("string"),
                    "chosen_conversations": [{"role": Value("string"), "content": Value("string")}],
                    "reject_conversations": [{"role": Value("string"), "content": Value("string")}],
                    "prompt": Value("string"),
                    "chosen": Value("string"),
                    "reject": Value("string"),
                    "preperence_ranking": [{"content": Value("string"), "ranking": Value("float")}],
                    "metadata": {
                        "main": Value("string"),
                        "middle": Value("string"),
                        "prompt_type": Value("string"),
                    },
                }
            )
        elif self.config.name == "PPO":
            features = Features(
                {
                    "id": Value("string"),
                    "prompt": Value("string"),
                    "metadata": {
                        "main": Value("string"),
                        "middle": Value("string"),
                        "prompt_type": Value("string"),
                    },
                }
            )

        return DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
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

        train_path_split = [path for path in zip_file_path if "Train" in path.as_posix()]
        valid_path_split = [path for path in zip_file_path if "Valid" in path.as_posix()]

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "file_path": train_path_split,
                    "split": "train",
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "file_path": valid_path_split,
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(self, file_path: List[Path], split) -> Generator:
        if self.config.name == "SFT":
            return self._generate_examples_sft(file_path, split)
        elif self.config.name == "RL":
            return self._generate_examples_rl(file_path, split)
        elif self.config.name == "PPO":
            return self._generate_examples_ppo(file_path, split)

    def _generate_examples_sft(self, file_path: List[Path], split: str) -> Generator:
        labeling_zip = [ZipFile(path) for path in file_path if "02.라벨링데이터" in path.as_posix()][0]
        sft_info = [zip_info for zip_info in labeling_zip.filelist if "/SFTlabel.json" == zip_info.filename][0]

        labels = json.loads(labeling_zip.open(sft_info).read().decode("utf-8"))

        for idx, example in enumerate(labels["data_info"]):
            conversations = [
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example["answer"]["contents"]},
            ]
            data = {
                "id": example["data_id"],
                "conversations": conversations,
                "prompt": example["question"],
                "answer": example["answer"]["contents"],
                "metadata": {
                    "main": example["data_category"]["main"],
                    "middle": example["data_category"]["middle"],
                    "prompt_type": example["question_type"],
                },
            }
            yield (idx, data)

    def _generate_examples_rl(self, file_path: List[Path], split: str) -> Generator:
        labeling_zip = [ZipFile(path) for path in file_path if "02.라벨링데이터" in path.as_posix()][0]
        rl_info = [zip_info for zip_info in labeling_zip.filelist if "/RMlabel.json" == zip_info.filename][0]

        labels = json.loads(labeling_zip.open(rl_info).read().decode("utf-8"))

        for idx, example in enumerate(labels["data_info"]):
            preperence_ls = [
                example["answer01"],
                example["answer02"],
                example["answer03"],
                example["answer04"],
                example["answer05"],
            ]
            preperence_ls = sorted(preperence_ls, key=lambda x: x["ranking"])

            for preperence in preperence_ls:
                preperence["content"] = preperence["contents"]
                del preperence["contents"]
                del preperence["answer_count"]

            chosen_conversations = [
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": preperence_ls[0]["content"]},
            ]
            reject_conversations = [
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": preperence_ls[-1]["content"]},
            ]

            data = {
                "id": example["data_id"],
                "chosen_conversations": chosen_conversations,
                "reject_conversations": reject_conversations,
                "prompt": example["question"],
                "chosen": preperence_ls[0]["content"],
                "reject": preperence_ls[-1]["content"],
                "preperence_ranking": preperence_ls,
                "metadata": {
                    "main": example["data_category"]["main"],
                    "middle": example["data_category"]["middle"],
                    "prompt_type": example["question_type"],
                },
            }
            yield (idx, data)

    def _generate_examples_ppo(self, file_path: List[Path], split: str) -> Generator:
        labeling_zip = [
            ZipFile(path) for path in file_path if "01.원천데이터" in path.as_posix() and "RLHF" in path.as_posix()
        ][0]
        ppo_info = [zip_info for zip_info in labeling_zip.filelist if "/PPOdata.json" == zip_info.filename][0]

        labels = json.loads(labeling_zip.open(ppo_info).read().decode("utf-8"))
        for idx, example in enumerate(labels["data_info"]):
            data = {
                "id": example["data_id"],
                "prompt": example["question"],
                "metadata": {
                    "main": example["data_category"]["main"],
                    "middle": example["data_category"]["middle"],
                    "prompt_type": example["question_type"],
                },
            }
            yield (idx, data)
