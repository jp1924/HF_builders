import io
import json
import os
import re
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
from PIL import Image as PIL_Image
from tqdm import tqdm


_LICENSE = """AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다."""


DATASET_KEY = "71709"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/0.5/{DATASET_KEY}.do"
_HOMEPAGE = f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"  # no lint


_DATANAME = "TableImageTextpairData"
DATASET_SIZE = 30.61

_DESCRIPTION = """- 표 이미지 및 이미지에 대응하는 내용 정보 텍스트를 쌍으로 구축하여 표 이미지의 내용 정보에 대한 요약문을 자동 생성하고, 표에 대한 다양한 관점을 제공하는 해설 문장을 생성하기 위함"""

bullet_header_regex = re.compile(r"  [0-9]\) |  [0-9]\. ")


class TableImageTextpairData(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(name="caption", version="1.0.0", description=_DESCRIPTION),
    ]

    DEFAULT_CONFIG_NAME = "caption"

    def _info(self) -> DatasetInfo:
        features = Features(
            {
                "id": Value("int32"),
                "image": Image(),
                "conversations": [
                    {"role": Value("string"), "content": Value("string")},
                ],
                "metadata": {
                    "doc_title": Value("string"),
                    "publisher": Value("string"),
                    "publish_year": Value("string"),
                    "table_type": Value("string"),
                    "table_field": Value("string"),
                    "table_unit": Value("string"),
                    "table_title": Value("string"),
                    "table_header": Value("string"),
                    "table_row_number": Value("int32"),
                    "table_column_number": Value("int32"),
                    "table_header_bold": Value("string"),
                    "table_background": Value("string"),
                    "html_path": Value("string"),
                    "width": Value("int32"),
                    "height": Value("int32"),
                },
                "summary": [Value("string")],
                "html": Value("string"),
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
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(self, filepath: List[Path], split: str) -> Generator:
        source_ls = [ZipFile(path) for path in filepath if "원천데이터" in path.as_posix()]
        source_ls = natsorted(source_ls, key=lambda x: x.filename)

        label_ls = [ZipFile(path) for path in filepath if "라벨링데이터" in path.as_posix()]
        label_ls = natsorted(label_ls, key=lambda x: x.filename)

        idx = 0
        for source_zip, label_zip in zip(source_ls, label_ls):
            source_info_ls = [file for file in source_zip.filelist if not file.is_dir()]
            source_info_ls = natsorted(source_info_ls, key=lambda x: x.filename)

            source_jpg_info_ls = [info for info in source_info_ls if "jpg" in info.filename]
            source_html_info_ls = [info for info in source_info_ls if "html" in info.filename]

            label_info_ls = [file for file in label_zip.filelist if not file.is_dir()]
            label_info_ls = natsorted(label_info_ls, key=lambda x: x.filename)

            if len(source_jpg_info_ls) != len(label_info_ls) != len(source_html_info_ls):
                raise ValueError("소스와 라벨 데이터의 개수가 틀림!")

            info_zip = zip(source_jpg_info_ls, source_html_info_ls, label_info_ls)
            for source_jpg_info, source_html_info, label_info in info_zip:
                label_contents = label_zip.open(label_info).read()
                html_contents = source_zip.open(source_html_info).read()
                source_contents = source_zip.open(source_jpg_info).read()

                label = json.loads(label_contents.decode("utf-8"))
                html = html_contents.decode("utf-8")
                image = source_contents

                # 'table_meta.table_background' = 'Y'
                # 'table_meta.html_path' = '/원천데이터/T01/C01/T01_C01_50000_1001_27.html'
                # 같이 되어 있음.

                pil_image = PIL_Image.open(io.BytesIO(image))
                metadata = {k.split(".")[1]: v for k, v in label["table_meta"].items()}
                metadata["width"] = pil_image.width
                metadata["height"] = pil_image.height

                # 1) **** 2) ***** 3)**** 이런식으로 작성되어 있음.
                text_explanation = label["table_data"]["table_data.text_explanation"]
                text_explanation = f"     {text_explanation}"

                text_explanation_ls = bullet_header_regex.split(text_explanation)
                text_explanation_ls = [text.strip() for text in text_explanation_ls if text.strip()]

                text_summary = label["table_data"]["table_data.text_summary"]
                text_summary = [text_summary] + text_explanation_ls

                # datasets features 특성 상 일관된 dtype을 가져야 하기 때문에 이렇게 해야 함.
                # 할꺼면 무조건 json.loads를 사용해야 함.
                conversations = [
                    {
                        "role": "user",
                        "content": json.dumps([{"type": "image"}]),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps([{"type": "text", "text": " ".join(text_summary)}], ensure_ascii=False),
                    },
                ]

                data = {
                    "id": label["file_id"],
                    "image": image,
                    "conversations": conversations,
                    "metadata": metadata,
                    "summary": text_summary,
                    "html": html,
                }

                yield (idx, data)
                idx += 1
