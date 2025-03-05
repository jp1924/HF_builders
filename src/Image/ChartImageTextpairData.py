import json
import os
import random
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
    Image,
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
차트 이미지에 대한 해석 데이터를 생성하며 차트 정보 추론이 가능한 서비스를 구축하는데 사용할 수 있는 인공지능 학습용 데이터 구축"""


DATASET_KEY = "71706"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/{DATASET_KEY}.do"
_HOMEPAGE = (
    f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"
)

_VERSION = "1.0.0"
_DATANAME = "ChartImageTextpairData"
DATASET_SIZE = 11.00


class ChartImageTextpairData(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(name="caption", version=_VERSION, description="캡션 데이터"),
    ]

    DEFAULT_CONFIG_NAME = "caption"

    def _info(self) -> DatasetInfo:
        features = Features(
            {
                "id": Value("int32"),
                "image": Image(),
                "conversations": [
                    {
                        "role": Value("string"),
                        "content": Value("string"),
                    },
                ],
                "metadata": {
                    "image_id": Value("int32"),
                    "data_category": Value("string"),
                    "chart_source": Value("string"),
                    "chart_color": Value("string"),
                    "chart_multi": Value("string"),
                    "chart_year": Value("string"),
                    "chart_main": Value("string"),
                    "chart_sub": Value("string"),
                    "width": Value("int32"),
                    "height": Value("int32"),
                    "annotations": [
                        {
                            "image_id": Value("int32"),
                            "is_title": Value("bool"),
                            "is_legend": Value("bool"),
                            "is_datalabel": Value("bool"),
                            "is_unit": Value("bool"),
                            "is_base": Value("bool"),
                            "is_axis_label_x_axis": Value("bool"),
                            "is_axis_label_y_axis": Value("bool"),
                            "title": Value("string"),
                            "legend": [Value("string")],
                            "unit": Value("string"),
                            "base": Value("string"),
                            "axis_title": {"x_axis": Value("string"), "y_axis": Value("string")},
                            "axis_label": {
                                "x_axis": [Value("string")],
                                "y_axis": [Value("string")],
                            },
                            "data_label": [
                                [Value("string")],
                            ],
                        }
                    ],
                },
                "summary": [Value("string")],
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

    def _generate_examples(self, filepath: List[Path], split: str) -> Generator:
        source_ls = [ZipFile(x) for x in filepath if "원천데이터" in str(x) and "이미지" in x.stem]
        source_ls = natsorted(source_ls, key=lambda x: x.filename)

        label_ls = [ZipFile(x) for x in filepath if "라벨링데이터" in str(x)]
        label_ls = natsorted(label_ls, key=lambda x: x.filename)

        idx = 0
        for source_zip, label_zip in zip(source_ls, label_ls):
            source_zip_file_info_ls = [x for x in source_zip.filelist if not x.is_dir()]
            source_zip_file_info_ls = natsorted(source_zip_file_info_ls, key=lambda x: x.filename)

            label_zip_file_info_ls = [x for x in label_zip.filelist if not x.is_dir()]
            label_zip_file_info_ls = natsorted(label_zip_file_info_ls, key=lambda x: x.filename)

            if len(source_zip_file_info_ls) != len(label_zip_file_info_ls):
                raise ValueError("소스와 라벨 데이터의 개수가 틀림!")

            for source_zip_file_info, label_zip_file_info in zip(source_zip_file_info_ls, label_zip_file_info_ls):
                label_contents = label_zip.open(label_zip_file_info).read()
                source_contents = source_zip.open(source_zip_file_info).read()

                label = json.loads(label_contents.decode("utf-8"))
                image = source_contents

                metadata = label["metadata"]
                metadata["width"] = label["image"][0]["width"]
                metadata["height"] = label["image"][0]["height"]
                metadata["annotations"] = label["annotations"]

                # datasets features 특성 상 일관된 dtype을 가져야 하기 때문에 이렇게 해야 함.
                # 할꺼면 무조건 json.loads를 사용해야 함.
                conversations = [
                    {
                        "role": "user",
                        "content": json.dumps([{"type": "image"}], ensure_ascii=False),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps([{"type": "text", "text": label["description"]}], ensure_ascii=False),
                    },
                ]

                data = {
                    "id": label["image"][0]["id"],
                    "image": image,
                    "conversations": conversations,
                    "metadata": metadata,
                    "summary": label["summary"],
                }

                yield (idx, data)
                idx += 1
