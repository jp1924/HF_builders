import json
import os
from pathlib import Path
from tarfile import TarFile
from typing import Generator, List
from zipfile import ZipFile

import requests
from datasets import BuilderConfig, DatasetInfo, Features, GeneratorBasedBuilder, Split, SplitGenerator, Value, Version
from datasets import logging as ds_logging
from kss import Kss
from natsort import natsorted
from tqdm import tqdm


_LICENSE = """
AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다."""
DESCRIPTION = """- 이름: 국가기록물 대상 초거대AI 학습을 위한 말뭉치 데이터
- 소개: 국가기록물 및 정부간행물을 활용한 초거대 AI 학습용 말뭉치 데이터셋 및 질의응답 데이터 구축. 
- 구축목적: 초거대 언어모델(LLM)을 기반으로 한 국가기록물 및 기타 문서 대상 질의응답 인공지능 모델 개발에 활용. 정부 데이터 통합 플랫폼 구축에 활용하여 정부와 민간의 데이터 접근성 향상 목적

# 데이터 구조
## 말뭉치
{
	"data": [
		{
			"source_id": "S0000001",
			"title": "제3기(2011~2014) 부천시지역사회복지계획",
			"publisher_company": "경기도",
			"category_main": "정치",
			"category_middle": "지방행정",
			"isbn": "CM00055107",
			"collection_name": "사업보고서",
			"contents_type": "공개",
			"issue_date": "2010",
            "corpus": " -{생략}- ",
        }
    ]
}
## 질의응답 데이터
{
	"Dataset": {
		"identifier": "unidocs",
		"name": "초거대AI 학습을 위한 말뭉치 데이터",
		"src_path": "dataSet/text/corpus",
		"label_path": "dataSet/text/label",
		"category": "2",
		"type": "0",
		"creation_date": "2023-07-03 12:00:00"
	},
	"data": [
		{
			"context_id": "C0000001",
			"raw_filename": "2010년도 남동구지역사회복지계획.pdf",
			"publisher": "인천광역시",
			"date": "2010",
			"type": {
				"collection_name": "사업보고서",
				"category_main": "정치",
				"category_middle": "지방행정"
			},
			"registration_no": "DM00007210",
			"source_id": "S0000007",
			"title": "2010년도 남동구지역사회복지계획",
			"context": " -{생략}- ",
			"labels": [
				{
					"label_id": "L0000001",
					"label_type": 2,
					"instructs": [
						{
							"instruct_id": "I0000001",
							"text": "존댓말+문어체",
							"meta": [
								{
									"category": "persona",
									"type": 1
								}
							]
						},
						{
							"instruct_id": "I0000002",
							"text": "지문을 읽고 질문에 맞는 답을 단답형 문장으로 답해줘",
							"meta": [
								{
									"category": "task",
									"type": 3
								}
							]
						},
						{
							"instruct_id": "I0000003",
							"text": "답변은 60자 이하로 답변해",
							"meta": [
								{
									"category": "output",
									"type": 1
								}
							]
						},
						{
							"instruct_id": "I0000004",
							"text": "지역복지 분야는 사업계획 목적의 명확화를 통해 사업의 전문성 확보가 필요하다고 평가되었습니까?",
							"meta": [
								{
									"category": "question",
									"type": 1
								}
							]
						}
					],
					"response": "네. 그렇습니다."
				}
			]
		},
    ]
}
# 폴더 구조
CorpusForLLMNationalRecordsAndArchives
 ┗ 119.국가기록물_대상_초거대AI_학습을_위한_말뭉치_데이터
 ┃ ┗ 3.개방데이터
 ┃ ┃ ┗ 1.데이터
 ┃ ┃ ┃ ┣ Training
 ┃ ┃ ┃ ┃ ┣ 01.원천데이터
 ┃ ┃ ┃ ┃ ┃ ┣ TS_1._말뭉치_데이터.zip
 ┃ ┃ ┃ ┃ ┃ ┃ ┣ BU_S0000001.json
 ┃ ┃ ┃ ┃ ┃ ┃ ┣ -{생략}- 
 ┃ ┃ ┃ ┃ ┗ 02.라벨링데이터
 ┃ ┃ ┃ ┃ ┃ ┣ TL_1._질의응답_데이터.zip
 ┃ ┃ ┃ ┃ ┃ ┃ ┣ training(yes,no).json
 ┃ ┃ ┃ ┃ ┃ ┃ ┗ training(의문사).json
 ┃ ┃ ┃ ┗ Validation
 ┃ ┃ ┃ ┃ ┣ 01.원천데이터
 ┃ ┃ ┃ ┃ ┃ ┣ VS_1._말뭉치_데이터.zip
 ┃ ┃ ┃ ┃ ┃ ┃ ┣ BU_S0000001.json
 ┃ ┃ ┃ ┃ ┃ ┃ ┣ -{생략}- 
 ┃ ┃ ┃ ┃ ┗ 02.라벨링데이터
 ┃ ┃ ┃ ┃ ┃ ┣ VL_1._말뭉치_데이터.zip
 ┃ ┃ ┃ ┃ ┃ ┃ ┣ training(yes,no).json
 ┃ ┃ ┃ ┃ ┃ ┃ ┗ training(의문사).json
와 같이 구성되어 있다. 그릐고 유해질의 데이터는 데이터 목적하고 맞지 않아서 따로 뺌.
"""

DATASET_KEY = "71788"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/0.6/{DATASET_KEY}.do"
_HOMEPAGE = (
    f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"
)

_DATANAME = "CorpusForLLMNationalRecordsAndArchives"
DATASET_SIZE = 0.8137

ds_logging.set_verbosity_info()
logger = ds_logging.get_logger("datasets")


class CorpusForLLMNationalRecordsAndArchives(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="CORPUS",
            data_dir="CORPUS",
            version="1.1.0",
            description="- 용도: 국가기록물로 구성된 말뭉치 데이터. " + DESCRIPTION,
        ),
        BuilderConfig(
            name="SFT",
            data_dir="SFT",
            version="1.1.0",
            description="- 용도: 국가기록물로 구성된 질의응답 데이터. " + DESCRIPTION,
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
                "metadata": {
                    "publisher_company": Value("string"),
                    "isbn": Value("string"),
                    "collection_name": Value("string"),
                    "contents_type": Value("string"),
                    "issue_date": Value("string"),
                    "category_main": Value("string"),
                    "category_middle": Value("string"),
                },
            }
        elif self.config.name == "SFT":
            features = {
                "id": Value("string"),
                "conversations": [
                    {
                        "role": Value("string"),
                        "content": Value("string"),
                        "persona": Value("string"),
                        "task": Value("string"),
                        "output": Value("string"),
                    },
                ],
                "prompt": Value("string"),
                "answer": Value("string"),
                "passage": Value("string"),
                "metadata": {
                    "publisher": Value("string"),
                    "date": Value("string"),
                    "collection_name": Value("string"),
                    "category_main": Value("string"),
                    "category_middle": Value("string"),
                    "registration_no": Value("string"),
                    "source_id": Value("string"),
                    "title": Value("string"),
                },
            }

        return DatasetInfo(
            description=self.config.description,
            version=Version(self.config.version),
            features=Features(features),
            homepage=_HOMEPAGE,
            license=_LICENSE,
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
        src_path_ls = filter(lambda path: "유해질의" not in path.as_posix(), src_path_ls)

        train_src_ls = filter(lambda path: "Training" in path.as_posix(), src_path_ls)
        valid_src_ls = filter(lambda path: "Validation" in path.as_posix(), src_path_ls)

        if self.config.name == "SFT":
            train_src_ls = [path for path in train_src_ls if "TL_1._질의응답_데이터.zip" in path.name]
            valid_src_ls = [path for path in valid_src_ls if "VL_1._말뭉치_데이터.zip" in path.name]  # 오타인듯
        elif self.config.name == "CORPUS":
            train_src_ls = [path for path in train_src_ls if "TS_1._말뭉치_데이터.zip" in path.name]
            valid_src_ls = [path for path in valid_src_ls if "VS_1._말뭉치_데이터.zip" in path.name]

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

    def _generate_examples(self, **kwargs) -> Generator:
        if self.config.name == "SFT":
            for idx, data in enumerate(self._sft_generate_examples(**kwargs)):
                data["id"] = idx
                yield (idx, data)
        elif self.config.name == "CORPUS":
            for idx, data in enumerate(self._corpus_generate_examples(**kwargs)):
                data["id"] = idx
                yield (idx, data)

    def _sft_generate_examples(self, file_ls: List[Path], split: str) -> Generator:
        zip = ZipFile(file_ls[0])

        for fileinfo in zip.infolist():
            raw_data_ls = json.load(zip.open(fileinfo))["data"]
            for raw_data in raw_data_ls:
                raw_chat = raw_data["labels"][0]
                instructs = raw_chat["instructs"]
                prompt, answer = instructs.pop()["text"], raw_chat["response"]
                persona, task, output = None, None, None
                for instruct in instructs:
                    category = instruct["meta"][0]["category"]
                    match category:
                        case "persona":
                            persona = instruct
                        case "task":
                            task = instruct
                        case "output":
                            output = instruct

                conversation = [
                    {
                        "role": "system",
                        "content": f"{persona['text']}로 {task['text']}. {output['text']}",
                        "persona": persona["text"],
                        "task": task["text"],
                        "output": output["text"],
                    },
                    {"role": "passage", "content": raw_data["context"]},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
                yield {
                    "id": raw_data["context_id"],
                    "conversations": conversation,
                    "prompt": prompt,
                    "answer": answer,
                    "passage": raw_data["context"],
                    "metadata": {
                        "publisher": raw_data["publisher"],
                        "date": raw_data["date"],
                        "collection_name": raw_data["type"]["collection_name"],
                        "category_main": raw_data["type"]["category_main"],
                        "category_middle": raw_data["type"]["category_middle"],
                        "registration_no": raw_data["registration_no"],
                        "source_id": raw_data["source_id"],
                        "title": raw_data["title"],
                        **raw_data["type"],
                    },
                }

    def _corpus_generate_examples(self, file_ls: List[Path], split: str) -> Generator:
        sentence_segmentator = Kss("split_sentences")
        zip = ZipFile(file_ls[0])

        for fileinfo in zip.infolist():
            raw_data_ls = json.load(zip.open(fileinfo))["data"]
            for raw_data in raw_data_ls:
                corpus = raw_data["corpus"]
                segment_ls = sentence_segmentator(corpus, backend="fast")
                clean_segment_ls = list()
                for segment in segment_ls:
                    segment = segment.strip()
                    if not segment:
                        continue
                    clean_segment_ls.append(segment)
                yield {
                    "id": raw_data["source_id"],
                    "corpus": corpus,
                    "sentence_ls": clean_segment_ls,
                    "category": f"{raw_data['category_main']}-{raw_data['category_middle']}",
                    "title": raw_data["title"],
                    "metadata": {
                        "publisher_company": raw_data["publisher_company"],
                        "isbn": raw_data["isbn"],
                        "collection_name": raw_data["collection_name"],
                        "contents_type": raw_data["contents_type"],
                        "issue_date": raw_data["issue_date"],
                        "category_main": raw_data["category_main"],
                        "category_middle": raw_data["category_middle"],
                    },
                }
