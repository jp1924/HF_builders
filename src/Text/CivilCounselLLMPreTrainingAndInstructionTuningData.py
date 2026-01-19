import json
import os
import random
from collections import defaultdict
from itertools import zip_longest
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
    Value,
    Version,
)
from datasets import logging as ds_logging
from natsort import natsorted
from tqdm import tqdm


_LICENSE = """AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다."""

DESCRIPTION = """"""

DATASET_KEY = "71844"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/0.5/{DATASET_KEY}.do"
_HOMEPAGE = f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"  # nolint

_DATANAME = "CivilCounselLLMPreTrainingAndInstructionTuningData"
DATASET_SIZE = 0.1664  # GB

ds_logging.set_verbosity_info()
logger = ds_logging.get_logger("datasets")


class CivilCounselLLMPreTrainingAndInstructionTuningData(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="CORPUS",
            data_dir="CORPUS",
            version="1.1.0",
            description="- 용도: 민사법으로 구성된 말뭉치 데이터. " + DESCRIPTION,
        ),
        BuilderConfig(
            name="SFT",
            data_dir="SFT",
            version="1.1.0",
            description="- 용도: 민사법으로 구성된 질의응답 데이터. " + DESCRIPTION,
        ),
        BuilderConfig(
            name="GRPO",
            data_dir="GRPO",
            version="1.1.0",
            description="- 용도: 민사법으로 구성된 강화학습 데이터 " + DESCRIPTION,
        ),
    ]
    DEFAULT_CONFIG_NAME = "SFT"
    DEFAULT_WRITER_BATCH_SIZE = 1000

    def _info(self):
        if self.config.name == "CORPUS":
            features = {
                "id": Value("string"),
                "corpus": Value("string"),
                "sentence_ls": [Value("string")],
                "category": Value("string"),
                "title": Value("string"),
            }
        elif self.config.name == "SFT":
            features = {
                "id": Value("string"),
                "conversations": [{"role": Value("string"), "content": Value("string")}],
                "prompt": Value("string"),
                "answer": Value("string"),
                "passage": Value("string"),
            }
        elif self.config.name == "GRPO":
            features = {
                "id": Value("int32"),
                "prompt": Value("string"),
                "choice": Value("string"),
                "answer": Value("string"),
                "passage": Value("string"),
                "conversations": [
                    {"role": Value("string"), "content": Value("string")},
                ],
            }

        return DatasetInfo(
            description=self.config.description,
            version=Version(self.config.version),
            features=Features(features),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=None,
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

    def _generate_examples(self, **kwargs):
        if self.config.name == "CORPUS":
            for idx, data in enumerate(self._corpus_generate_examples(**kwargs)):
                yield idx, data
        elif self.config.name == "SFT":
            for idx, data in enumerate(self._sft_generate_examples(**kwargs)):
                yield idx, data
        elif self.config.name == "GRPO":
            for idx, data in enumerate(self._grpo_generate_examples(**kwargs)):
                yield idx, data

    def _corpus_generate_examples(self, file_ls: List[Path], split: str) -> Generator:
        src_path_ls = filter(lambda path: "원천데이터" in str(path), file_ls)
        src_path_ls = natsorted(src_path_ls, key=lambda src_path: src_path.stem)

        breakpoint()

        id_counter = 0
        for src_path in src_path_ls:
            src_zip = ZipFile(src_path, "r")
            src_info_ls = filter(lambda info: not info.is_dir(), src_zip.infolist())

            for src_info in src_info_ls:
                source = json.load(src_zip.open(src_info))

                yield {
                    "id": str(id_counter),
                    "corpus": "".join(source["sentences"]).strip(),
                    "sentence_ls": source["sentences"],
                    "category": None,
                    "title": None,
                }

                id_counter += 1

    def _sft_generate_examples(self, file_ls: List[Path], split: str) -> Generator:
        lbl_path_ls = filter(lambda path: "라벨링데이터" in str(path), file_ls)
        lbl_path_ls = natsorted(lbl_path_ls, key=lambda lbl_path: lbl_path.stem)

        id_counter = 0
        for lbl_path in lbl_path_ls:
            lbl_zip = ZipFile(lbl_path, "r")
            lbl_info_ls = filter(lambda info: not info.is_dir(), lbl_zip.infolist())

            for lbl_info in lbl_info_ls:
                labels = json.load(lbl_zip.open(lbl_info))

                taskinfo = labels["taskinfo"]
                info = labels["info"]
                info["instruction_case"] = taskinfo["instruction_case"]

                system = taskinfo["instruction"] if "input" in taskinfo else None
                prompt = taskinfo["input"] if "input" in taskinfo else taskinfo["instruction"]
                answer = taskinfo["output"]

                conversations = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
                if system:
                    conversations = [{"role": "system", "content": system}] + conversations

                yield {
                    "id": str(id_counter),
                    "conversations": conversations,
                    "prompt": prompt,
                    "system": system,
                    "answer": answer,
                    "passage": "".join(taskinfo["sentences"]),
                }

                id_counter += 1

    def _grpo_generate_examples(self, file_ls: List[Path], split: str) -> Generator:
        lbl_path_ls = filter(lambda path: "라벨링데이터" in str(path), file_ls)
        lbl_path_ls = natsorted(lbl_path_ls, key=lambda lbl_path: lbl_path.stem)
        id_counter = 0

        doc_class_map = defaultdict(list)
        for lbl_path in lbl_path_ls:
            lbl_zip = ZipFile(lbl_path, "r")
            lbl_info_ls = filter(lambda info: not info.is_dir(), lbl_zip.infolist())

            for lbl_info in lbl_info_ls:
                labels = json.load(lbl_zip.open(lbl_info))
                if "input" not in labels["taskinfo"]:
                    continue

                doc_class_map[labels["info"]["doc_class"]].append(labels)

        chunk_ls = list()
        for labels_ls in doc_class_map.values():
            random.shuffle(labels_ls)
            chunk_ls.extend(list(filter(lambda x: None not in x, zip_longest(*[iter(labels_ls)] * 4, fillvalue=None))))

        for chunk in chunk_ls:
            if len(chunk) < 4:
                continue

            # 각 샘플에서 질문/답변/문단 추출
            samples = list(chunk)
            try:
                prompt_ls = [s["taskinfo"]["input"].strip() for s in samples]
                answer_raw_ls = [s["taskinfo"]["output"].strip() for s in samples]
                passage_ls = ["".join(s["taskinfo"]["sentences"]).strip() for s in samples]
            except KeyError:
                # 형식 불일치 시 스킵
                continue

            # (prompt, answer, passage) 튜플 리스트
            chat_ls = list(zip(prompt_ls, answer_raw_ls, passage_ls))
            random.shuffle(chat_ls)  # 보기 섞기
            shuffled_passages = passage_ls[:]  # passage는 독립적으로 다시 섞음
            random.shuffle(shuffled_passages)

            # 선택지 문자열 구성 (예: [1] 답변내용)
            # 필요하면 형식 변경 가능: 여기서는 단순히 기존 로직 재현
            choice_items = [f"[{idx}] {ans.strip()}" for idx, (_, ans, _) in enumerate(chat_ls, 1)]
            choices_str = "\n".join(choice_items)

            # 질문 하나 선택
            question, _, _ = random.choice(chat_ls)
            # 정답 인덱스 (1-based)
            correct_answer_idx = next(idx for idx, (p, _, _) in enumerate(chat_ls, 1) if p == question)

            passage_merged = "\n".join(shuffled_passages)

            yield {
                "id": str(id_counter),
                "prompt": question,
                "choice": choices_str,
                "answer": str(correct_answer_idx),
                "passage": passage_merged,
                "conversations": [
                    {"role": "passage", "content": passage_merged},
                    {"role": "user", "content": f"{question}\n\n{choices_str}"},
                ],
            }

            id_counter += 1
