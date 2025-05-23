import json
import os
import random
from pathlib import Path
from tarfile import TarFile
from typing import List
from zipfile import ZipFile

import requests
from datasets import (
    BuilderConfig,
    DatasetInfo,
    Features,
    GeneratorBasedBuilder,
    Image,
    Split,
    SplitGenerator,
    Value,
)
from natsort import natsorted
from tqdm import tqdm


_LICENSE = """AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다."""

_DESCRIPTION = """이미지와 이미지에 대한 질문과 대답으로 구성된 시각정보 기반 질의응답(Visual Question Answering, VQA) 데이터셋을 구축하여 시각정보 기반 질의응답 기술 연구의 학습용 데이터셋으로 활용 가능한 이미지 데이터 제공"""


DATASET_KEY = "104"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/0.5/{DATASET_KEY}.do"
_HOMEPAGE = f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"  # no lint


_DATANAME = "VisualQuestionAnswering"
DATASET_SIZE = 85.03


class VisualQuestionAnswering(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [BuilderConfig(name="default", version="1.0.0", description=_DESCRIPTION)]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        features = Features(
            {
                "id": Value("string"),
                "image": Image(),
                "category": Value("string"),
                "question_answer": [
                    {
                        "id": Value("string"),
                        "question": Value("string"),
                        "answer": Value("string"),
                        "confidence": Value("bool"),
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
                    "split": "validlidation",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        random.seed(42)
        source_ls = [ZipFile(x) for x in filepath if "원천데이터" in str(x)]
        label_ls = [ZipFile(x) for x in filepath if "라벨링데이터" in str(x)]
        info_dict = {
            info.filename.split("/")[-1]: info for info in sum([zip_file.filelist for zip_file in source_ls], [])
        }
        source_dict = {zip_file.filename.split("/")[-1].replace(".zip", ""): zip_file for zip_file in source_ls}

        category_dict = dict()
        for x in label_ls[0].filelist:
            category = x.filename.split("/")[0]
            if category not in category_dict:
                category_dict[category] = []
            category_dict[category].append(x)

        idx = 0
        for category, file_ls in category_dict.items():
            file_ls = [file for file in file_ls if not file.is_dir()]

            label_dict = dict()
            for file in file_ls:
                _type = file.filename.split("/")[1]

                if _type not in label_dict:
                    label_dict[_type] = []

                label_dict[_type].append(file)
            breakpoint()
            for _type, label_file in label_dict.items():
                annotation = [x for x in label_file if "annotation.json" in x.filename][0]
                images = [x for x in label_file if "images.json" in x.filename][0]
                question = [x for x in label_file if "question.json" in x.filename][0]

                annotation_ls = json.loads(label_ls[0].open(annotation).read().decode("utf-8"))["annotations"]
                images_ls = json.loads(label_ls[0].open(images).read().decode("utf-8"))["images"]
                question_ls = json.loads(label_ls[0].open(question).read().decode("utf-8"))["questions"]

                annotation_dict = dict()
                for x in annotation_ls:
                    if x["image_id"] not in annotation_dict:
                        annotation_dict[x["image_id"]] = []

                    annotation_dict[x["image_id"]].append(x)
                question_dict = dict()
                for x in question_ls:
                    if x["image_id"] not in question_dict:
                        question_dict[x["image_id"]] = []

                    question_dict[x["image_id"]].append(x)

                if not (len(annotation_dict) == len(question_dict) == len(images_ls)):
                    raise ValueError("(len(annotation_dict) == len(question_dict) == len(images_ls)) 갈아 언멎음")

                for image_row in images_ls:
                    annotation = annotation_dict[image_row["image_id"]]
                    question = question_dict[image_row["image_id"]]

                    annotation = natsorted(annotation, key=lambda x: x["question_id"])
                    question = natsorted(question, key=lambda x: x["question_id"])

                    if len(annotation) != len(question):
                        raise ValueError("len(annotation) != len(question) 길이 안맞음.")

                    question_answer = list()
                    for x, y in zip(question, annotation):
                        question_answer.append(
                            {
                                "id": x["question_id"],
                                "question": x["question"],
                                "answer": y["multiple_choice_answer"],
                                "confidence": True if y["answer_confidence"] == "yes" else False,
                            }
                        )

                    data = {
                        "id": image_row["image_id"],
                        "image": source_dict[image_row["category"]].open(info_dict[image_row["image"]]).read(),
                        "category": image_row["category"],
                        "question_answer": question_answer,
                    }

                    yield (idx, data)
                    idx += 1
