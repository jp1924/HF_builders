import json
import os
from pathlib import Path
from tarfile import TarFile
from zipfile import ZipFile

import datasets
import requests
from datasets import Features, Image, Value
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


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class OutsideKnowledgebasedMultimodalQAData(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")

    def _info(self):
        features = Features(
            {
                "IMAGE_ID": Value("string"),
                "IMAGE_NAME": Value("string"),
                "KOREAN_IMAGE": Value("string"),
                "DIRECT_IMAGE": Value("string"),
                "ACTION": Value("string"),
                "CAPTION": Value("string"),
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
                "bounding_box": [
                    {
                        "BOX_ID": Value("int32"),
                        "OBJECT": Value("string"),
                        "X_COORDINATE": Value("int32"),
                        "Y_COORDINATE": Value("int32"),
                        "BOX_WIDTH": Value("int32"),
                        "BOX_HEIGHT": Value("int32"),
                    }
                ],
                "questions": [
                    {
                        "question_id": Value("string"),
                        "visual_concept": [Value("string")],
                        "question_ko": Value("string"),
                        "question_en": Value("string"),
                        "answer_ko": Value("string"),
                        "answer_en": Value("string"),
                        "question_type": [Value("string")],
                        "ans_source": Value("string"),
                        "kb_source": [Value("string")],
                        "fact": [Value("string")],
                    }
                ],
                "IMAGE": Image(),
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

    def aihub_downloader(self, recv_path: Path):
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

    def unzip_data(self, tar_file: Path, unzip_dir: Path) -> list:
        with TarFile(tar_file, "r") as mytar:
            mytar.extractall(unzip_dir)
            os.remove(tar_file)

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

        return list(unzip_dir.rglob("*.zip*"))

    def _split_generators(self, dl_manager):
        data_name = "Outside_Knowledge_based_Multimodal_QA_Data"
        cache_dir = Path(dl_manager.download_config.cache_dir)
        unzip_dir = cache_dir.joinpath(data_name)

        if not unzip_dir.exists():
            tar_file = cache_dir.joinpath(f"{data_name}.tar")
            self.aihub_downloader(tar_file)
            # 압축이 덜 출렸을 때를 고려해야 함.
            zip_file_path = self.unzip_data(tar_file, unzip_dir)
        else:
            zip_file_path = list(unzip_dir.rglob("*.zip"))

        train_split = [x for x in zip_file_path if "Training" in str(x)]
        valid_split = [x for x in zip_file_path if "Validation" in str(x)]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_split,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": valid_split,
                    "split": "valid",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        source_ls = [ZipFile(x) for x in filepath if "원천데이터" in str(x)]
        label_ls = [ZipFile(x) for x in filepath if "라벨링데이터" in str(x) and "VQA" in str(x)]

        label = json.loads(label_ls[0].open(label_ls[0].filelist[0]).read().decode("utf-8"))
        label = {x["IMAGE_NAME"]: x for x in label["annotations"]}
        for zip_file in source_ls:
            for file in zip_file.filelist:
                filename = file.filename.replace("/", "")
                label[filename]["IMAGE"] = zip_file.open(file.filename).read()
                image_id = label[filename]["IMAGE_ID"]

                yield (int(image_id), label[filename])
