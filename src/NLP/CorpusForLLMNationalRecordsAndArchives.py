import json
import os
from collections import defaultdict
from pathlib import Path
from tarfile import TarFile
from typing import Generator, List
from zipfile import ZipFile

import requests
from datasets import BuilderConfig, DatasetInfo, Features, GeneratorBasedBuilder, Split, SplitGenerator, Value, Version
from kss import Kss
from natsort import natsorted
from tqdm import tqdm


_LICENSE = """
AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다.
"""
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
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/{DATASET_KEY}.do"
_HOMEPAGE = (
    f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"
)

_DATANAME = "CorpusForLLMNationalRecordsAndArchives"
DATASET_SIZE = 0.83317


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
                "conversation": [
                    {
                        "role": Value("string"),
                        "content": Value("string"),
                        "persona": Value("string"),
                        "task": Value("string"),
                        "output": Value("string"),
                    },
                    {"role": Value("string"), "content": Value("string")},
                    {"role": Value("string"), "content": Value("string")},
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
        part_dict = defaultdict(list)
        for part_path in unzip_dir.rglob("*.zip.part*"):
            parh_stem = str(part_path.parent.joinpath(part_path.stem))
            part_dict[parh_stem].append(part_path)

        for dst_path, part_path_ls in part_dict.items():
            with open(dst_path, "wb") as byte_f:
                for part_path in natsorted(part_path_ls):
                    byte_f.write(part_path.read_bytes())
                    os.remove(part_path)

    def _split_generators(self, dl_manager) -> List[SplitGenerator]:  # type: ignore
        def get_zip_path_ls() -> List[Path]:
            cache_dir = Path(dl_manager.download_config.cache_dir)

            unzip_dir = cache_dir.joinpath(_DATANAME)
            tar_file = cache_dir.joinpath(f"{_DATANAME}.tar")

            if not unzip_dir.exists():
                self.aihub_downloader(tar_file)

                with TarFile(tar_file, "r") as mytar:
                    mytar.extractall(unzip_dir)
                self.concat_zip_part(unzip_dir)
                os.remove(tar_file)

            zip_path_ls = list()
            exclude_ls = ["유해질의"]
            for zip_path in unzip_dir.rglob("*.zip"):
                if any(exclude in zip_path.as_posix() for exclude in exclude_ls):
                    continue
                zip_path_ls.append(zip_path)
            return zip_path_ls

        zip_path_ls = get_zip_path_ls()

        if self.config.name == "SFT":
            train_zip_path = [zip_path for zip_path in zip_path_ls if zip_path.name == "TL_1._질의응답_데이터.zip"][0]
            valid_zip_path = [zip_path for zip_path in zip_path_ls if zip_path.name == "VL_1._말뭉치_데이터.zip"][0]
            split_generator_ls = [
                SplitGenerator(
                    name=Split.TRAIN,
                    gen_kwargs={
                        "zip_path": train_zip_path,
                        "split": "train",
                    },
                ),
                SplitGenerator(
                    name=Split.VALIDATION,
                    gen_kwargs={
                        "zip_path": valid_zip_path,
                        "split": "validation",
                    },
                ),
            ]
        elif self.config.name == "CORPUS":
            train_zip_path = [zip_path for zip_path in zip_path_ls if zip_path.name == "TS_1._말뭉치_데이터.zip"][0]
            valid_zip_path = [zip_path for zip_path in zip_path_ls if zip_path.name == "VS_1._말뭉치_데이터.zip"][0]
            split_generator_ls = [
                SplitGenerator(
                    name=Split.TRAIN,
                    gen_kwargs={
                        "zip_path": train_zip_path,
                        "split": "train",
                    },
                ),
                SplitGenerator(
                    name=Split.VALIDATION,
                    gen_kwargs={
                        "zip_path": valid_zip_path,
                        "split": "validation",
                    },
                ),
            ]
        return split_generator_ls

    def _generate_examples(self, zip_path: Path, split: str) -> Generator:
        if self.config.name == "SFT":
            for idx, data in enumerate(self._generate_examples_sft(zip_path, split)):
                yield (idx, data)
        elif self.config.name == "CORPUS":
            for idx, data in enumerate(self._generate_examples_corpus(zip_path, split)):
                yield (idx, data)

    def _generate_examples_sft(self, zip_path: Path, split: str) -> Generator:
        zip = ZipFile(zip_path)

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
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
                yield {
                    "id": raw_data["context_id"],
                    "conversation": conversation,
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

    def _generate_examples_corpus(self, zip_path: Path, split: str) -> Generator:
        sentence_segmentator = Kss("split_sentences")
        zip = ZipFile(zip_path)

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
