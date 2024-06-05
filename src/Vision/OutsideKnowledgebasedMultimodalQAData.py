import json
import os
from pathlib import Path
from tarfile import TarFile
from typing import List
from zipfile import ZipFile

import datasets
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
    Version,
)
from natsort import natsorted
from tqdm import tqdm


_LICENSE = """
AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다.
"""

_DESCRIPTION = """\
인간이 가진 상식적인 지식이나 배경지식을 바탕으로, 이미지에 관련한 질문에 대해 이미지 속에서 답을 찾아야 하는 태스크
"""

DATASET_KEY = "71357"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/{DATASET_KEY}.do"
_HOMEPAGE = (
    f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"
)
_VERSION = Version("1.1.0")
_DATANAME = "OutsideKnowledgebasedMultimodalQAData"
DATASET_SIZE = 11.55


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class OutsideKnowledgebasedMultimodalQAData(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [BuilderConfig(name="default", version=_VERSION)]
    DEFAULT_CONFIG_NAME = "default"
    VERSION = _VERSION

    def _info(self) -> DatasetInfo:
        features = Features(
            {
                "id": Value("string"),
                "image": Image(),
                "caption": Value("string"),
                "caption_ls": [Value("string")],
                "category": Value("string"),
                "objects": [
                    {
                        "id": Value("int32"),
                        "text": Value("string"),
                        "bbox": [Value("int32")],
                    }
                ],
                "question_answer": [
                    {
                        "id": Value("string"),
                        "question": Value("string"),
                        "answer": Value("string"),
                        "visual_concept": [Value("string")],
                        "question_en": Value("string"),
                        "answer_en": Value("string"),
                        "question_type": [Value("string")],
                        "ans_source": Value("string"),
                        "kb_source": [Value("string")],
                        "fact": [Value("string")],
                    }
                ],
                "metadata": {
                    "KOREAN_IMAGE": Value("string"),
                    "DIRECT_IMAGE": Value("string"),
                    "ACTION": Value("string"),
                    "IMAGE_ACQUISITION_DATE": Value("string"),
                    "IMAGE_URL": Value("string"),
                    "LICENSE": Value("string"),
                    "MAINOBJECT": Value("string"),
                    "MAINOBJECT_URL": Value("string"),
                    "SCENE": Value("string"),
                    "SUBOBJECT_1": Value("string"),
                    "SUBOBJECT_2": Value("string"),
                    "WIDTH": Value("string"),
                    "HEIGHT": Value("string"),
                    "WIKI_FILE": Value("string"),
                },
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=None,
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

    def _generate_examples(self, filepath, split):
        source_ls = [ZipFile(x) for x in filepath if "원천데이터" in str(x)]
        label_ls = [ZipFile(x) for x in filepath if "라벨링데이터" in str(x) and "VQA" in str(x)][0]

        label_ls = label_ls.open(label_ls.filelist[0])
        label = json.loads(label_ls.read().decode("utf-8"))
        label = {x["IMAGE_NAME"]: x for x in label["annotations"]}

        image_id = 0
        for zip_file in source_ls:
            for file in zip_file.filelist:
                filename = file.filename.replace("/", "")
                label_file = label[filename]

                question_answer_ls = list()
                for row in label_file["questions"]:
                    new_row = {
                        "id": row["question_id"],
                        "question": row["question_ko"],
                        "answer": row["answer_ko"],
                        "visual_concept": row["visual_concept"],
                        "question_en": row["question_en"],
                        "answer_en": row["answer_en"],
                        "question_type": row["question_type"],
                        "ans_source": row["ans_source"],
                        "kb_source": row["kb_source"],
                        "fact": row["fact"],
                    }

                    question_answer_ls.append(new_row)
                objects_ls = list()
                for row in label_file["bounding_box"]:
                    new_row = {
                        "id": row["BOX_ID"],
                        "text": row["OBJECT"],
                        "bbox": [
                            row["X_COORDINATE"],
                            row["Y_COORDINATE"],
                            row["X_COORDINATE"] + row["BOX_WIDTH"],
                            row["Y_COORDINATE"] + row["BOX_HEIGHT"],
                        ],
                    }

                    objects_ls.append(new_row)
                data = {
                    "id": label_file["IMAGE_ID"],
                    "image": zip_file.open(file.filename).read(),
                    "caption": label_file["CAPTION"],
                    "caption_ls": [label_file["CAPTION"]],
                    "category": label_file["MAINOBJECT"],
                    "objects": objects_ls,
                    "question_answer": question_answer_ls,
                    "metadata": {
                        "KOREAN_IMAGE": label_file["KOREAN_IMAGE"],
                        "DIRECT_IMAGE": label_file["DIRECT_IMAGE"],
                        "ACTION": label_file["ACTION"],
                        "IMAGE_ACQUISITION_DATE": label_file["IMAGE_ACQUISITION_DATE"],
                        "IMAGE_URL": label_file["IMAGE_URL"],
                        "LICENSE": label_file["LICENSE"],
                        "MAINOBJECT": label_file["MAINOBJECT"],
                        "MAINOBJECT_URL": label_file["MAINOBJECT_URL"],
                        "SCENE": label_file["SCENE"],
                        "SUBOBJECT_1": label_file["SUBOBJECT_1"],
                        "SUBOBJECT_2": label_file["SUBOBJECT_2"],
                        "WIDTH": label_file["WIDTH"],
                        "HEIGHT": label_file["HEIGHT"],
                        "WIKI_FILE": label_file["WIKI_FILE"],
                    },
                }

                yield (int(image_id), data)

                image_id += 1
