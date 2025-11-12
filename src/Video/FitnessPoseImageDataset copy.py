import json
import os
from collections import defaultdict
from pathlib import Path
from tarfile import TarFile
from typing import Generator, List, Tuple
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
from datasets import logging as ds_logging
from natsort import natsorted
from tqdm import tqdm


_LICENSE = """AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다."""


DATASET_KEY = "231"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/0.5/{DATASET_KEY}.do"
_HOMEPAGE = f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"  # nolint


_DATANAME = "FitnessPoseImageDataset"
DATASET_SIZE = 1062.8397  # GB
SAMPLE_RATE = 16000


_DESCRIPTION = """2017년 - 2020년 개방 데이터
데이터 변경이력
|   버전 | 일자       | 변경내용         | 비고   |
|-------:|:-----------|:-----------------|:-------|
|    1.2 | 2022-02-07 | 데이터 품질 보완 |        |
|    1.1 | 2022-01-11 | 데이터 추가 개방 |        |
|    1   | 2021-06-18 | 데이터 최초 개방 |        |
데이터 히스토리
| 일자       | 변경내용             | 비고   |
|:-----------|:---------------------|:-------|
| 2024-05-20 | 저작도구 매뉴얼 변경 |        |
소개
다양한 자세와 체형을 가신 사람들로부터 홈트레이닝(운동자세 분석 및 추천), AR/MR 서비스(5G 기반 증강현실 컨텐츠), 피트니스 수집 플랫폼(머신러닝 모델 적용), 의료분야(재활치료 및 재활운동 자세 교정, 원격치료) 등과 같은 세밀한 행동을 인식할 수 있는 AI모델 개발을 통해 데이터 인프라 구축
- 데이터 영역 : 헬스케어
- 데이터 유형 : 이미지
- 구축년도 : 2020년
- 구축량 : 200,000만 클립
구축목적
피트니스 데이터 구축을 통해서 Untact시대에 적합한 홈트레이닝 서비스를 만들 수 있는 기초를 만들고, 국가차원의 인공지능 경쟁력을 강화시키기 위함

2017년 - 2020년 개방 데이터
데이터 변경이력
|   버전 | 일자       | 변경내용         | 비고   |
|-------:|:-----------|:-----------------|:-------|
|    1.2 | 2022-02-07 | 데이터 품질 보완 |        |
|    1.1 | 2022-01-11 | 데이터 추가 개방 |        |
|    1   | 2021-06-18 | 데이터 최초 개방 |        |
데이터 히스토리
| 일자       | 변경내용             | 비고   |
|:-----------|:---------------------|:-------|
| 2024-05-20 | 저작도구 매뉴얼 변경 |        |
소개
다양한 자세와 체형을 가신 사람들로부터 홈트레이닝(운동자세 분석 및 추천), AR/MR 서비스(5G 기반 증강현실 컨텐츠), 피트니스 수집 플랫폼(머신러닝 모델 적용), 의료분야(재활치료 및 재활운동 자세 교정, 원격치료) 등과 같은 세밀한 행동을 인식할 수 있는 AI모델 개발을 통해 데이터 인프라 구축
- 데이터 영역 : 헬스케어
- 데이터 유형 : 이미지
- 구축년도 : 2020년
- 구축량 : 200,000만 클립
구축목적
피트니스 데이터 구축을 통해서 Untact시대에 적합한 홈트레이닝 서비스를 만들 수 있는 기초를 만들고, 국가차원의 인공지능 경쟁력을 강화시키기 위함

구축 내용 및 제공 데이터량
| 동작                        | 수집피사체                        | 촬영 Clip                                           | 기본 정보                 | 주요 특징                        |
|:----------------------------|:----------------------------------|:----------------------------------------------------|:--------------------------|:---------------------------------|
|                             |                                   |                                                     |                           |                                  |
| 30종의 동작(5개의 운동상태) | 성별, 체형, 키크기 등 다양한 형태 | 200,000 만 Clip (5개 Multiview로 40,000만 Clip수집) | COCO 17 Skeleton keypoint | 정자세, 오류자세를 구분하여 취득 |

설명서 및 활용가이드 다운로드
데이터 설명서 다운로드
구축활용가이드 다운로드
※
이 데이터에 포함된 인물의 얼굴 등에 대해서는 개인정보 및 초상권의 이용 동의를 받아 제공합니다
.
데이터 변경이력
|   버전 | 일자       | 변경내용         | 비고   |
|-------:|:-----------|:-----------------|:-------|
|    1.2 | 2022.02.07 | 데이터 품질 보완 |        |
|    1.1 | 2022.01.11 | 데이터 추가 개방 |        |
|    1   | 2021.06.18 | 데이터 최초 개방 |        |
구축목적
피트니스 데이터 구축을 통해서 Untact시대에 적합한 홈트레이닝 서비스를 만들 수 있는 기초를 만들고, 국가차원의 인공지능 경쟁력을 강화시키기 위함
활용 분야
홈트레이닝(운동자세 분석 및 추천), AR/MR 서비스(5G 기반 증강현실 컨텐츠), 피트니스 수집 플랫폼(머신러닝 모델 적용), 의료분야(재활치료 및 재활운동 자세 교정, 원격치료) 등
소개
다양한 자세와 체형을 가진 사람들로부터 3D human pose를 capture하는 것을 넘어서 사람의 다양하고 세밀한 운동 종류 및 상태를 추가적으로 capture하는 것을 통해, 일상생활의 행동을 인식할 뿐만 아니라 운동 동작 및 자세 등과 같은 세밀한 행동을 인식할 수 있는 AI모델을 개발하여 데이터 인프라를 구축하고자 함
구축 내용 및 제공 데이터량
| 동작                        | 수집피사체                        | 촬영 Clip                                           | 기본 정보                 | 주요 특징                        |
|:----------------------------|:----------------------------------|:----------------------------------------------------|:--------------------------|:---------------------------------|
|                             |                                   |                                                     |                           |                                  |
| 30종의 동작(5개의 운동상태) | 성별, 체형, 키크기 등 다양한 형태 | 200,000 만 Clip (5개 Multiview로 40,000만 Clip수집) | COCO 17 Skeleton keypoint | 정자세, 오류자세를 구분하여 취득 |
대표도면
< 데이터 구축 도면 >
< AI모델링 도면 >
필요성
인간 자세 인식은 로봇과 인간의 상호작용과 CCTV를 통한 범죄 및 위험상황 감시를 위해 반드시 필요한 연구분야이며, 많은 컴퓨터 비젼 연구자들에 의해서 오랫동안 연구되어 왔음, 다양성이 높은 비디오들을 모두 포함하는 큰 규모의 데이터 셋 취득의 어려움 때문에 꼭 연구되어야 하는 분야임, 뿐만 아니라, 최근 Untact 시대의 도래로, 집에서도 운동 동작에 대한 정확한 코칭이 가능한 서비스에 대한 수요가 증대되는 만큼 해당 데이터셋은 무궁무진한 활용 가능성을 품고 있음
데이터 구조
총 30가지 운동, 운동 별 32개의 운동상태가 정의되어 960개의 Status표현
| 구분              | 설명                   | 비고                       |
|:------------------|:-----------------------|:---------------------------|
|                   |                        |                            |
| Annotation  No1.  | 운동상태 정의          | 운동 마다 5개의 운동 상태) |
| Annotation  No2.. | 3D Keypoint Coordinate |                            |"""

ds_logging.set_verbosity_info()
logger = ds_logging.get_logger("datasets")


class FitnessPoseImageDataset(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        # BuilderConfig(name="A_video", version="1.2.0", description="좌측 135도 지점" + _DESCRIPTION),
        # BuilderConfig(name="B_video", version="1.2.0", description="좌측 45도 지점" + _DESCRIPTION),
        BuilderConfig(name="C_video", version="1.2.0", description="정면 0도 지점" + _DESCRIPTION),
        # BuilderConfig(name="D_video", version="1.2.0", description="우측 45도 지점" + _DESCRIPTION),
        # BuilderConfig(name="E_video", version="1.2.0", description="우측 135도 지점" + _DESCRIPTION),
    ]

    DEFAULT_CONFIG_NAME = "C_video"
    DEFAULT_WRITER_BATCH_SIZE = 100

    def _info(self) -> DatasetInfo:
        features = Features(
            {
                "id": Value("string"),
                "image_ls": [Image()],
                "point_ls": [
                    {
                        "Nose": {"x": Value("int32"), "y": Value("int32")},
                        "Left Eye": {"x": Value("int32"), "y": Value("int32")},
                        "Right Eye": {"x": Value("int32"), "y": Value("int32")},
                        "Left Ear": {"x": Value("int32"), "y": Value("int32")},
                        "Right Ear": {"x": Value("int32"), "y": Value("int32")},
                        "Left Shoulder": {"x": Value("int32"), "y": Value("int32")},
                        "Right Shoulder": {"x": Value("int32"), "y": Value("int32")},
                        "Left Elbow": {"x": Value("int32"), "y": Value("int32")},
                        "Right Elbow": {"x": Value("int32"), "y": Value("int32")},
                        "Left Wrist": {"x": Value("int32"), "y": Value("int32")},
                        "Right Wrist": {"x": Value("int32"), "y": Value("int32")},
                        "Left Hip": {"x": Value("int32"), "y": Value("int32")},
                        "Right Hip": {"x": Value("int32"), "y": Value("int32")},
                        "Left Knee": {"x": Value("int32"), "y": Value("int32")},
                        "Right Knee": {"x": Value("int32"), "y": Value("int32")},
                        "Left Ankle": {"x": Value("int32"), "y": Value("int32")},
                        "Right Ankle": {"x": Value("int32"), "y": Value("int32")},
                        "Neck": {"x": Value("int32"), "y": Value("int32")},
                        "Left Palm": {"x": Value("int32"), "y": Value("int32")},
                        "Right Palm": {"x": Value("int32"), "y": Value("int32")},
                        "Back": {"x": Value("int32"), "y": Value("int32")},
                        "Waist": {"x": Value("int32"), "y": Value("int32")},
                        "Left Foot": {"x": Value("int32"), "y": Value("int32")},
                        "Right Foot": {"x": Value("int32"), "y": Value("int32")},
                        "frame": Value("int32"),
                    }
                ],
                "metadata": {
                    "key": Value("string"),
                    "type": Value("string"),
                    "pose": Value("string"),
                    "exercise": Value("string"),
                    "conditions": [
                        {"condition": Value("string"), "value": Value("bool")},
                    ],
                    "description": Value("string"),
                    "info": Value("string"),
                    "width": Value("int32"),
                    "height": Value("int32"),
                },
            }
        )

        return DatasetInfo(
            description=self.config.description,
            version=self.config.version,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    # Modify _aihub_downloader to extract tar files only once
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
            train_lbl_ls = list((data_dir / "013.피트니스자세" / "1.Training" / "라벨링데이터").rglob("*.zip"))
            valid_lbl_ls = list((data_dir / "013.피트니스자세" / "2.Validation" / "라벨링데이터").rglob("*.zip"))

            train_src_ls = list((data_dir / "013.피트니스자세" / "1.Training" / "원시데이터").glob("*"))
            valid_src_ls = list((data_dir / "013.피트니스자세" / "2.Validation" / "원시데이터").glob("*"))

            return train_lbl_ls + valid_lbl_ls + train_src_ls + valid_src_ls

        aihub_api_key = os.getenv("AIHUB_API_KEY", None)
        if not aihub_api_key:
            raise ValueError(
                """AIHUB_API_KEY가 지정되지 않았습니다. `os.environ["AIHUB_API_KEY"]="your_key"`로 ID를 지정해 주세요"""
            )

        download_from_aihub(download_path, aihub_api_key)
        unzip_tar_file(download_path, data_dir)
        concat_zip_part(data_dir)

        # 얘만 특별히
        logger.info("tar 파일로 압축되어 있는 데이터는 제작 시 속도가 너무 느린 문제가 있음.")
        for path in tqdm(list(data_dir.rglob("*.tar"))):
            print(f"Unzipping {path}")
            unzip_tar_file(path, path.parent / path.stem)

        lbl_dir = data_dir / "013.피트니스자세" / "1.Training" / "라벨링데이터"
        find_lbl_zip = {path.name for path in lbl_dir.rglob("*.zip")}
        select_lbl = {"기구_Labeling.zip", "맨몸운동_Labeling_new_220128.zip", "바벨_덤벨_Labeling_new_220128.zip"}
        # NOTE: 기구_Labeling.zip 이 라벨 파일 안에 01, 02, 03, 04, 05의 라벨 zip 파일로 또 나뉘어져 있어서 이렇게 압축 해제가 필요로 하다.
        delete_lbl_ls = find_lbl_zip - select_lbl
        for delete_lbl in delete_lbl_ls:
            delete_lbl_path = lbl_dir / delete_lbl
            if delete_lbl_path.exists():
                logger.info(f"Deleting {delete_lbl_path}")
                delete_lbl_path.unlink()

        for lbl_name in select_lbl:
            print(f"Unzipping {lbl_name}")
            # 여기서 lbl zip 파일 압축 해제
            lbl_zip_path = lbl_dir / lbl_name
            if lbl_zip_path.exists():
                logger.info(f"Unzipping {lbl_zip_path}")
                with ZipFile(lbl_zip_path, "r") as lbl_zip:
                    lbl_zip.extractall(lbl_dir / lbl_name.replace(".zip", ""))

                lbl_zip_path.unlink()

        msg = "dataset builder에서 데이터 다시 다운받을지 말지를 결정하는 파일이다. 이거 지우면 aihub에서 데이터 다시 다운 받음."
        complete_file_path.write_text(msg)

        train_lbl_ls = list((data_dir / "013.피트니스자세" / "1.Training" / "라벨링데이터").rglob("*.zip"))
        valid_lbl_ls = list((data_dir / "013.피트니스자세" / "2.Validation" / "라벨링데이터").rglob("*.zip"))

        train_src_ls = list((data_dir / "013.피트니스자세" / "1.Training" / "원시데이터").glob("*"))
        valid_src_ls = list((data_dir / "013.피트니스자세" / "2.Validation" / "원시데이터").glob("*"))

        return train_lbl_ls + valid_lbl_ls + train_src_ls + valid_src_ls

    def _split_generators(self, dl_manager) -> List[SplitGenerator]:  # type: ignore
        cache_dir = Path(dl_manager.download_config.cache_dir)

        download_path = cache_dir.joinpath(f"{_DATANAME}.tar")
        # NOTE: 일부가 tar파일로 되어 있기 때문에 rglob("*.zip") -> rglob("*.tar")로 변경함.
        src_path_ls = self._aihub_downloader(download_path)

        train_src_ls = [path for path in src_path_ls if "1.Training" in path.parts]
        valid_src_ls = [path for path in src_path_ls if "2.Validation" in path.parts]

        split_generator_ls = [
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "file_ls": valid_src_ls,
                    "split": "validation",
                },
            ),
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "file_ls": train_src_ls,
                    "split": "train",
                },
            ),
        ]

        return split_generator_ls

    def _generate_examples(self, **kwagrs) -> Generator:
        for idx, data in enumerate(self._x_video_generate_examples(**kwagrs)):
            yield idx, data

    def _x_video_generate_examples(self, file_ls: List[Path], split: str):
        def get_korean_name(path) -> str:
            """Convert English exercise type to Korean name for sorting"""
            type_map = (
                {"furniture": "기구", "babel": "바벨_덤벨", "body": "맨몸운동"}
                if split == "train"
                else {"valid_wheel_data": "furniture_01", "valid_babel_data": "babel_01", "valid_body_data": "body_01"}
            )
            for english, korean in type_map.items():
                name = path.name
                if english in name:
                    return name.replace(english, korean)
            raise ValueError(f"Unknown type in path: {path}")

        def get_jpeg_size(jpeg_bytes: bytes) -> Tuple[int, int]:
            # JPEG 파일은 0xFFD8로 시작
            if jpeg_bytes[0:2] != b"\xff\xd8":
                raise ValueError("Not a JPEG file")

            idx = 2
            while idx < len(jpeg_bytes):
                # 마커 찾기
                while jpeg_bytes[idx] != 0xFF:
                    idx += 1
                while jpeg_bytes[idx] == 0xFF:
                    idx += 1
                marker = jpeg_bytes[idx]
                idx += 1

                # SOF0(0xC0), SOF2(0xC2) 등에서 크기 정보가 있음
                if marker in [0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF]:
                    length = int.from_bytes(jpeg_bytes[idx : idx + 2], "big")
                    idx += 2
                    precision = jpeg_bytes[idx]
                    idx += 1
                    height = int.from_bytes(jpeg_bytes[idx : idx + 2], "big")
                    idx += 2
                    width = int.from_bytes(jpeg_bytes[idx : idx + 2], "big")
                    return width, height
                else:
                    # 다른 마커는 건너뜀
                    length = int.from_bytes(jpeg_bytes[idx : idx + 2], "big")
                    idx += length
            raise ValueError("No size found in JPEG")

        lbl_zip_ls = natsorted(
            [ZipFile(lbl_path) for lbl_path in filter(lambda path: "라벨링데이터" in path.parts, file_ls)],
            key=lambda zip: zip.filename,
        )
        src_dir_ls = natsorted(filter(lambda path: "원시데이터" in path.parts, file_ls), key=get_korean_name)

        if len(lbl_zip_ls) != len(src_dir_ls):
            raise ValueError(
                "라벨링 데이터와 원시 데이터의 개수가 같아야 합니다. "
                "라벨링 데이터와 원시 데이터의 개수를 확인해 주세요."
            )

        side_view_map = {"A": "view1", "B": "view2", "C": "view3", "D": "view4", "E": "view5"}
        view_point = side_view_map[self.config.name.split("_")[0]]

        for lbl_zip, src_dir in zip(lbl_zip_ls, src_dir_ls):
            if "body" not in src_dir.as_posix():
                logger.warning(f"\nSkipping {lbl_zip.filename} as it is not a body label")
                continue

            # Extract label files
            lbl_info_ls = natsorted(
                filter(lambda info: not info.is_dir() and "3d" not in info.filename, lbl_zip.infolist()),
                key=lambda info: info.filename,
            )
            src_path_ls = natsorted(src_dir.rglob("*.jpg"))

            video_img_map = defaultdict(list)
            for src_path in src_path_ls:
                video_img_map[src_path.parent.stem].append(src_path)

            logger.info(f"\ndata-size: {len(video_img_map)}, label-size: {len(lbl_info_ls)}")
            for lbl_info in lbl_info_ls:
                try:
                    labels = json.load(lbl_zip.open(lbl_info))
                    img_path_ls = [frame[view_point]["img_key"] for frame in labels["frames"]]
                    frame_key = {Path(img_path).parent.stem for img_path in img_path_ls}.pop()
                    if frame_key not in video_img_map:
                        raise ValueError(
                            f"Frame key {frame_key} not found in video image map. "
                            "Check if the label file matches the source images."
                        )
                    img_path_ls = [path.read_bytes() for path in video_img_map[frame_key]]
                    img_size_ls = [get_jpeg_size(byte) for byte in img_path_ls]

                    if len(set(img_size_ls)) != 1:
                        raise ValueError(
                            f"Image sizes are not consistent for {lbl_info.filename}. "
                            "All images must have the same dimensions."
                        )

                    for i, frame in enumerate(labels["frames"]):
                        view = frame[view_point]
                        pts = view["pts"]
                        width, height = img_size_ls[i]
                        for name, coord in pts.items():
                            x, y = coord["x"], coord["y"]
                            if not (0 <= x < width and 0 <= y < height):
                                raise ValueError(
                                    f"Out of bounds: frame {i}, keypoint {name}, ({x},{y}) not in ({width},{height})"
                                )

                except BaseException as e:
                    logger.warning(f"Error: {e} in {lbl_info.filename}")
                    continue

                point_ls = list()
                for frame in labels["frames"]:
                    view = frame[view_point]
                    img_key, _, pts = view.pop("img_key"), view.pop("active"), view.pop("pts")
                    point_ls.append(
                        {
                            **pts,
                            "frame": int(img_key.split("-")[-1].replace(".jpg", "")),
                            "img_key": img_key,
                        }
                    )

                width, height = set(img_size_ls).pop()
                metadata = labels["type_info"]
                metadata["info"] = str({Path(frame["img_key"]).parent for frame in point_ls}.pop())
                metadata["width"] = width
                metadata["height"] = height

                yield {
                    "id": frame_key,
                    "image_ls": img_path_ls,
                    "point_ls": point_ls,
                    "metadata": metadata,
                }
