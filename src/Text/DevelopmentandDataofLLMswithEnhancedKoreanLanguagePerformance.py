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

from transformers import set_seed


set_seed(42)

_LICENSE = """AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다."""

_CITATION = None

_DESCRIPTION = """한국어 말뭉치 데이터 20억 어절/310만 건과 Reinforcement Learning Human Feedback(RLHF) 데이터 7만 7천 건으로 구성"""


DATASET_KEY = "71748"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/0.5/{DATASET_KEY}.do"
_HOMEPAGE = f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"  # no lint


_DATANAME = "DevelopmentandDataofLLMswithEnhancedKoreanLanguagePerformance"
DATASET_SIZE = 16.07

ds_logging.set_verbosity_info()
logger = ds_logging.get_logger("datasets")


class DevelopmentandDataofLLMswithEnhancedKoreanLanguagePerformance(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(name="SFT", version="1.1.0", data_dir="SFT"),
        BuilderConfig(name="RL", version="1.1.0", data_dir="RL"),
        BuilderConfig(name="PPO", version="1.1.0", data_dir="PPO"),
    ]
    DEFAULT_CONFIG_NAME = "SFT"

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

        train_path_split = [path for path in src_path_ls if "Train" in path.as_posix()]
        valid_path_split = [path for path in src_path_ls if "Valid" in path.as_posix()]

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

    def _generate_examples(self, **kwargs) -> Generator:
        if self.config.name == "SFT":
            for idx, data in enumerate(self._generate_examples_sft(**kwargs)):
                yield idx, data
        elif self.config.name == "RL":
            for idx, data in enumerate(self._generate_examples_rl(**kwargs)):
                yield idx, data
        elif self.config.name == "PPO":
            for idx, data in enumerate(self._generate_examples_ppo(**kwargs)):
                yield idx, data

    def _generate_examples_sft(self, file_path: List[Path], split: str) -> Generator:
        labeling_zip = [ZipFile(path) for path in file_path if "02.라벨링데이터" in path.as_posix()][0]
        sft_info = [zip_info for zip_info in labeling_zip.filelist if "/SFTlabel.json" == zip_info.filename][0]

        labels = json.loads(labeling_zip.open(sft_info).read().decode("utf-8"))

        for idx, example in enumerate(labels["data_info"]):
            prompt, answer = example["question"], example["answer"]["contents"]
            conversations = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]

            if "<NAME>" in prompt or "<NAME>" in answer:
                continue

            if answer == prompt:
                continue

            yield {
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

            chosen, reject, prompt = preperence_ls[0]["content"], preperence_ls[-1]["content"], example["question"]

            chosen_conversations = [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}]
            reject_conversations = [{"role": "user", "content": prompt}, {"role": "assistant", "content": reject}]

            if "<NAME>" in chosen or "<NAME>" in reject or "<NAME>" in prompt:
                continue

            if chosen == prompt or reject == prompt:
                continue

            yield {
                "id": example["data_id"],
                "chosen_conversations": chosen_conversations,
                "reject_conversations": reject_conversations,
                "prompt": prompt,
                "chosen": chosen,
                "reject": reject,
                "preperence_ranking": preperence_ls,
                "metadata": {
                    "main": example["data_category"]["main"],
                    "middle": example["data_category"]["middle"],
                    "prompt_type": example["question_type"],
                },
            }

    def _generate_examples_ppo(self, file_path: List[Path], split: str) -> Generator:
        labeling_zip = [
            ZipFile(path) for path in file_path if "01.원천데이터" in path.as_posix() and "RLHF" in path.as_posix()
        ][0]
        ppo_info = [zip_info for zip_info in labeling_zip.filelist if "/PPOdata.json" == zip_info.filename][0]

        labels = json.loads(labeling_zip.open(ppo_info).read().decode("utf-8"))
        for idx, example in enumerate(labels["data_info"]):
            if "<NAME>" in example["question"]:
                continue

            yield {
                "id": example["data_id"],
                "prompt": example["question"],
                "metadata": {
                    "main": example["data_category"]["main"],
                    "middle": example["data_category"]["middle"],
                    "prompt_type": example["question_type"],
                },
            }
