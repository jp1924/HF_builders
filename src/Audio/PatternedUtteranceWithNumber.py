import io
import json
import os
import re
import wave
from itertools import zip_longest
from pathlib import Path
from tarfile import TarFile
from typing import Generator, List, Literal
from unicodedata import normalize
from zipfile import ZipFile

import requests
import torch
from datasets import Audio, BuilderConfig, DatasetInfo, Features, GeneratorBasedBuilder, Split, SplitGenerator, Value
from datasets import logging as ds_logging
from natsort import natsorted
from setproctitle import setproctitle
from torch.multiprocessing import Manager, Queue, spawn
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import pipeline


_LICENSE = """AI 데이터 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.

본 AI 데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 다만 기존의 데이터셋을 활용하시어 만들어진 2차 저작물(훈련으로 만들어진 지능형 제품・서비스, 챗봇 등) 은 영리적・비영리적 활용이 가능합니다."""


DATASET_KEY = "484"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/0.5/{DATASET_KEY}.do"
_HOMEPAGE = f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"  # nolint


_DATANAME = "PatternedUtteranceWithNumber"
DATASET_SIZE = 958.365234375  # GB
SAMPLE_RATE = 16000


_DESCRIPTION = """# 모델 메타데이터
2017년 - 2020년 개방 데이터
데이터 변경이력
|   버전 | 일자       | 변경내용          | 비고   |
|-------:|:-----------|:------------------|:-------|
|    1.1 | 2023-06-28 | 라벨링데이터 수정 |        |
|    1   | 2022-07-12 | 데이터 최초 개방  |        |
데이터 히스토리
| 일자       | 변경내용          | 비고   |
|:-----------|:------------------|:-------|
| 2025-05-08 | 구축업체정보 수정 |        |
| 2022-07-12 | 콘텐츠 최초 등록  |        |
소개
본 데이터는 한자어, 고유어, 외래어 등의 숫자 읽기 다양성을 반영하여 84개의 카테고리로 구성된 10,000시간 이상의 음성데이터로 구성하였음. (스크립트 데이터 포함)
구축목적
다양한 환경의 발화 특성을 반영한 음성 데이터를 구축하여 음성인식 기반 AI서비스의 확대, 발전의 토대를 마련

2017년 - 2020년 개방 데이터
데이터 변경이력
|   버전 | 일자       | 변경내용          | 비고   |
|-------:|:-----------|:------------------|:-------|
|    1.1 | 2023-06-28 | 라벨링데이터 수정 |        |
|    1   | 2022-07-12 | 데이터 최초 개방  |        |
데이터 히스토리
| 일자       | 변경내용          | 비고   |
|:-----------|:------------------|:-------|
| 2025-05-08 | 구축업체정보 수정 |        |
| 2022-07-12 | 콘텐츠 최초 등록  |        |
소개
본 데이터는 한자어, 고유어, 외래어 등의 숫자 읽기 다양성을 반영하여 84개의 카테고리로 구성된 10,000시간 이상의 음성데이터로 구성하였음. (스크립트 데이터 포함)
구축목적
다양한 환경의 발화 특성을 반영한 음성 데이터를 구축하여 음성인식 기반 AI서비스의 확대, 발전의 토대를 마련

1. 데이터 구축 규모
| 데이터 구분   | 데이터 구분.1   | 세부내역                                                        | 구축규모                         |
|:--------------|:----------------|:----------------------------------------------------------------|:---------------------------------|
| 데이터셋      | 스크립트        | 세부항목 약 84개의 카테고리로 구성된 스크립트 데이터 구축       | 149,998건                        |
| 데이터셋      | 음성            | 발화자, 발화 환경에 따른 864가지의 구축 환경을 통한 데이터 가공 | 13,342시간 (묵음제외 11,842시간) |
2. 데이터 분포 : 고유어, 한자어, 외래어 읽기 특성에 따른 84가지 카테고리의 숫자 데이터 구축
| 읽기 특성      | 패턴          |   가짓수 |   스크립트수 |   발화시간(초) |
|:---------------|:--------------|---------:|-------------:|---------------:|
| 한자어         | 통계/수치     |        8 |        17100 |        5233261 |
| 한자어         | 날짜/시간     |        5 |        11300 |        3447005 |
| 한자어         | 통화/금액     |        3 |        15400 |        4125304 |
| 한자어         | 교통정보      |        3 |         6000 |        1415068 |
| 한자어         | 나이/생년월일 |        2 |         5500 |        1674635 |
| 한자어         | 신분증번호    |        4 |         7700 |        2751724 |
| 한자어         | 주소/구역     |        5 |         9000 |        2852654 |
| 한자어         | 사이즈        |        2 |         3600 |        1091626 |
| 한자어         | 단위          |        7 |         7500 |        2709001 |
| 한자어         | 금융/은행     |        6 |        15800 |        5554667 |
| 한자어         | 주문정보      |        3 |         5100 |        1520145 |
| 한자어         | 헬스케어      |        3 |         3300 |        1218988 |
| 한자어         | 스포츠        |        5 |         1500 |         512193 |
| 한자어         | 자동생성번호  |        3 |         5400 |        1901728 |
| 한자어         | 통신번호      |        5 |         8998 |        3205763 |
| 한자어         | 개인고유번호  |        3 |         2700 |         917338 |
| 한자어         | 사업자번호    |        1 |         1800 |         741246 |
| 고유어/ 외래어 | 기수          |       12 |        19900 |        6470781 |
| 고유어/ 외래어 | 서수          |        2 |         1200 |         336003 |
| 고유어/ 외래어 | 사투리        |        1 |          200 |          61554 |
| 고유어/ 외래어 | 외래어        |        1 |         1000 |         292196 |
3. 구축 환경 : 발화자, 녹음환경을 고려한 864가지의 환경을 통해 데이터 구축
| 구축기준   | 분류기준               | 구축비율   |
|:-----------|:-----------------------|:-----------|
| 성별       | 남성 음성 발화 데이터  | 40%        |
| 성별       | 여성 음성 발화 데이터  | 60%        |
| 지역       | 수도권                 | 76%        |
| 지역       | 경상도                 | 6%         |
| 지역       | 전라도                 | 6%         |
| 지역       | 충청도                 | 6%         |
| 지역       | 강원도                 | 5%         |
| 지역       | 제주도                 | 1%         |
| 연령대     | 20대 미만              | 2%         |
| 연령대     | 20대                   | 15%        |
| 연령대     | 30대                   | 26%        |
| 연령대     | 40대                   | 27%        |
| 연령대     | 50대                   | 27%        |
| 연령대     | 60대 이상              | 3%         |
| 녹음환경   | 클린환경 (~ 40dB)      | 15%        |
| 녹음환경   | 일반환경 (41dB ~ 60dB) | 75%        |
| 녹음환경   | 소음환경 (61dB ~)      | 10%        |
| 녹음기기   | 휴대폰 마이크          | 72%        |
| 녹음기기   | 유선 이어폰            | 13%        |
| 녹음기기   | 무선 이어폰            | 13%        |
| 녹음기기   | AI스피커               | 2%         |
<구축 기준별 참조 이미지>

설명서 및 활용가이드 다운로드
데이터 설명서 다운로드
구축활용가이드 다운로드
1. 데이터 포맷
유형별 데이터 포맷
| 데이터 구분   | 데이터 유형   | 데이터 포맷   | 비고                                        |
|:--------------|:--------------|:--------------|:--------------------------------------------|
| 라벨데이터    | 스크립트      | JSON          | 스크립트 데이터를 전사하여 음성 데이터 구축 |
| 원천데이터    | 스크립트      | txt           | 스크립트 데이터를 전사하여 음성 데이터 구축 |
| 가공데이터    | 음성          | pcm           | 스크립트 데이터를 전사하여 음성 데이터 구축 |
2. 데이터 구성
라벨링 데이터
| 분류         | 순서   | 속성표기            | 속성명             | 속성 설명                                              | 데이터 타입   | 필수 여부   |
|:-------------|:-------|:--------------------|:-------------------|:-------------------------------------------------------|:--------------|:------------|
| 분류         | 순서   | 속성표기            | 속성명             | 속성 설명                                              | 데이터 타입   | 필수 여부   |
|              |        |                     |                    |                                                        |               |             |
| 녹음발화정보 | 1      | recordedID          | 녹음ID             | 녹음관리번호                                           | String        | O           |
| 녹음발화정보 | 2      | recordedDate        | 녹음일시           | 녹음한 날짜                                            | String        | O           |
| 녹음발화정보 | 3      | recordedStart       | 녹음시작시간       | 녹음 스타트 시간                                       | String        | O           |
| 녹음발화정보 | 4      | fileName            | 녹음파일명         | 녹음 음성파일명                                        | String        | O           |
| 녹음발화정보 | 5      | filePath            | 녹음파일위치       | 녹음 음성파일 위치                                     | String        | O           |
| 녹음발화정보 | 6      | recordedTime        | 녹음파일재생시간   | 녹음음성파일 재생시간(초단위)                          | Numeric       | O           |
| 녹음발화정보 | 7      | recordQuality       | 음질               | 음성파일의 음질 구분                                   | String        | O           |
| 녹음발화정보 | 8      | recordedDevice      | 녹음 수집 디바이스 | 음성파일 녹음된 디바이스                               | String        | O           |
| 녹음발화정보 | 9      | redcordedDeviceName | 녹음 수집          | 음성파일 녹음 된                                       | String        | X           |
| 녹음발화정보 | 9      | redcordedDeviceName | 디바이스           | 디바이스 회사명                                        | String        | X           |
| 녹음발화정보 | 9      | redcordedDeviceName | 회사명             |                                                        | String        | X           |
| 녹음발화정보 | 10     | recordedEnv         | 녹음 환경          | 구축된 데이터의 활용 용처별 구분을 위한 녹음 환경 구분 | String        | O           |
| 녹음자정보   | 11     | collectedID         | 수집방법ID         | 크라우드소싱등 구분ID                                  | String        | O           |
| 녹음자정보   | 12     | recorderID          | 녹음자ID           | 녹음자ID                                               | String        | O           |
| 녹음자정보   | 13     | sex                 | 성별               | 발화자의 성별                                          | String        | O           |
| 녹음자정보   | 14     | generation          | 세대               | 발화자의 세대                                          | String        | O           |
| 녹음자정보   | 15     | residence           | 거주지역           | 발화자의 거주지역(광역시, 도별)                        | String        | O           |
| 녹음자정보   | 16     | dialect             | 화자방언여부       | 발화자 음성의 사투리 여부                              | String        | O           |
| 녹음자정보   | 17     | dialectRegion       | 화자방언지역       | 발화자의 음성이 방언인 경우, 방언 지역                 | String        | O           |
| 녹음자정보   | 18     | areaInfomation      | 발화자의           | 이전 거주지역 과 부모님의 고향 정보                    | String        | O           |
| 녹음자정보   | 18     | areaInfomation      | 지역정보           | 이전 거주지역 과 부모님의 고향 정보                    | String        | O           |
원천 데이터
| 분류         | 순서   | 속성표기         | 속성명                      | 속성 설명                                            | 데이터 타입   | 필수 여부   |
|:-------------|:-------|:-----------------|:----------------------------|:-----------------------------------------------------|:--------------|:------------|
| 분류         | 순서   | 속성표기         | 속성명                      | 속성 설명                                            | 데이터 타입   | 필수 여부   |
|              |        |                  |                             |                                                      |               |             |
| 녹음대화정보 | 1      | scriptID         | 스크립트ID                  | 스크립트ID                                           | String        | O           |
| 녹음대화정보 | 1      | scriptID         | 스크립트ID                  | 스크립트ID                                           | String        | O           |
| 녹음대화정보 | 2      | scriptITN        | 음성파일 전사 TEXT(ITN표기) | 영문/숫자를 영어와 아라비아 숫자 글자 그대로 표시    | String        | O           |
| 녹음대화정보 | 3      | scriptTN         | 음성파일 전사 TEXT(TN표기)  | 영문/숫자를 영어와 아라비아 숫자를 발음방법대로 표시 | String        | O           |
| 녹음대화정보 | 4      | scriptNumberWord | 숫자가 포함된 단어          | 단어별 인식을 위한 추출                              | String        | O           |
| 녹음대화정보 | 5      | patternTheme     | 패턴 주제                   | 숫자포함 패턴 문장, 숫자의 패턴 주제별 다양성 구분   | String        | O           |

# 트리 구조
└─159.숫자가 포함된 패턴 발화 데이터
    └─01.데이터
        ├─1.Training
        │  ├─라벨링데이터
        │  │  ├─TL_13.신분증번호.zip | 255 MB | 34050
        │  │  ├─TL_12.스포츠.zip | 53 MB | 34049
        │  │  ├─TL_11.사투리.zip | 7 MB | 34048
        │  │  ├─TL_5.금융-은행.zip | 551 MB | 34062
        │  │  ├─TL_4.교통정보.zip | 160 MB | 34061
        │  │  ├─TL_3.고유어서수.zip | 41 MB | 34060
        │  │  ├─TL_21.헬스케어.zip | 124 MB | 34059
        │  │  ├─TL_20.통화-금액.zip | 462 MB | 34058
        │  │  ├─TL_19.통신번호.zip | 351 MB | 34056
        │  │  ├─TL_17.주소-구역.zip | 320 MB | 34054
        │  │  ├─TL_16.주문정보.zip | 167 MB | 34053
        │  │  ├─TL_7.날짜-시간.zip | 381 MB | 34064
        │  │  ├─TL_6.나이-생년월일.zip | 202 MB | 34063
        │  │  ├─TL_9.사업자번호.zip | 72 MB | 34066
        │  │  ├─TL_8.단위.zip | 276 MB | 34065
        │  │  ├─TL_15.자동생성번호.zip | 183 MB | 34052
        │  │  ├─TL_14.외래영어.zip | 34 MB | 34051
        │  │  ├─TL_18.통계-수치.zip | 555 MB | 34055
        │  │  ├─TL_10.사이즈.zip | 135 MB | 34047
        │  │  └─TL_1.개인고유번호.zip | 107 MB | 34046
        │  ├─원천데이터
        │  │  ├─TS_1.개인고유번호(음성).zip | 19 GB | 34067
        │  │  ├─TS_1.개인고유번호(텍스트).zip | 1 MB | 34068
        │  │  ├─TS_10.사이즈(음성).zip | 22 GB | 34069
        │  │  ├─TS_10.사이즈(텍스트).zip | 2 MB | 34070
        │  │  ├─TS_11.사투리(음성).zip | 1 GB | 34071
        │  │  ├─TS_11.사투리(텍스트).zip | 83 KB | 34072
        │  │  ├─TS_12.스포츠(음성).zip | 10 GB | 34073
        │  │  ├─TS_12.스포츠(텍스트).zip | 661 KB | 34074
        │  │  ├─TS_13.신분증번호(음성).zip | 55 GB | 34075
        │  │  ├─TS_13.신분증번호(텍스트).zip | 4 MB | 34076
        │  │  ├─TS_14.외래영어(음성).zip | 6 GB | 34077
        │  │  ├─TS_14.외래영어(텍스트).zip | 458 KB | 34078
        │  │  ├─TS_15.자동생성번호(음성).zip | 39 GB | 34079
        │  │  ├─TS_15.자동생성번호(텍스트).zip | 2 MB | 34080
        │  │  ├─TS_16.주문정보(음성).zip | 31 GB | 34081
        │  │  ├─TS_16.주문정보(텍스트).zip | 2 MB | 34082
        │  │  ├─TS_17.주소-구역(음성).zip | 57 GB | 34083
        │  │  ├─TS_17.주소-구역(텍스트).zip | 4 MB | 34084
        │  │  ├─TS_18.통계-수치(음성)_1.zip | 57 GB | 34085
        │  │  ├─TS_18.통계-수치(음성)_2.zip | 50 GB | 34086
        │  │  ├─TS_18.통계-수치(텍스트).zip | 8 MB | 34087
        │  │  ├─TS_19.통신번호(음성).zip | 66 GB | 34088
        │  │  ├─TS_19.통신번호(텍스트).zip | 4 MB | 34089
        │  │  ├─TS_2.고유어기수(음성)_1.zip | 66 GB | 34090
        │  │  ├─TS_2.고유어기수(음성)_2.zip | 64 GB | 34091
        │  │  ├─TS_2.고유어기수(텍스트).zip | 8 MB | 34092
        │  │  ├─TS_20.통화-금액(음성).zip | 84 GB | 34093
        │  │  ├─TS_20.통화-금액(텍스트).zip | 7 MB | 34094
        │  │  ├─TS_21.헬스케어(음성).zip | 24 GB | 34095
        │  │  ├─TS_21.헬스케어(텍스트).zip | 1 MB | 34096
        │  │  ├─TS_3.고유어서수(음성).zip | 7 GB | 34097
        │  │  ├─TS_3.고유어서수(텍스트).zip | 499 KB | 34098
        │  │  ├─TS_4.교통정보(음성).zip | 29 GB | 34099
        │  │  ├─TS_4.교통정보(텍스트).zip | 3 MB | 34100
        │  │  ├─TS_5.금융-은행(음성)_1.zip | 36 GB | 34101
        │  │  ├─TS_5.금융-은행(음성)_2.zip | 78 GB | 34102
        │  │  ├─TS_5.금융-은행(텍스트).zip | 7 MB | 34103
        │  │  ├─TS_6.나이-생년월일(음성).zip | 35 GB | 34104
        │  │  ├─TS_6.나이-생년월일(텍스트).zip | 2 MB | 34105
        │  │  ├─TS_7.날짜-시간(음성).zip | 70 GB | 34106
        │  │  ├─TS_7.날짜-시간(텍스트).zip | 5 MB | 34107
        │  │  ├─TS_8.단위(음성).zip | 55 GB | 34108
        │  │  ├─TS_8.단위(텍스트).zip | 3 MB | 34109
        │  │  ├─TS_9.사업자번호(텍스트).zip | 860 KB | 34110
        │  │  └─TS_9.사업자번호.zip | 15 GB | 34111
        │  └─라벨링데이터_20230628_add
        │      └─TL_2.고유어기수_보완.zip | 739 MB | 246634
        └─2.Validation
            ├─라벨링데이터
            │  ├─VL_1.개인고유번호.zip | 12 MB | 34112
            │  ├─VL_10.사이즈.zip | 15 MB | 34113
            │  ├─VL_11.사투리.zip | 916 KB | 34114
            │  ├─VL_12.스포츠.zip | 6 MB | 34115
            │  ├─VL_13.신분증번호.zip | 31 MB | 34116
            │  ├─VL_14.외래영어.zip | 4 MB | 34117
            │  ├─VL_15.자동생성번호.zip | 22 MB | 34118
            │  ├─VL_16.주문정보.zip | 19 MB | 34119
            │  ├─VL_17.주소-구역.zip | 38 MB | 34120
            │  ├─VL_18.통계-수치.zip | 67 MB | 34121
            │  ├─VL_19.통신번호.zip | 42 MB | 34122
            │  ├─VL_20.통화-금액.zip | 56 MB | 34124
            │  ├─VL_21.헬스케어.zip | 16 MB | 34125
            │  ├─VL_3.고유어서수.zip | 4 MB | 34126
            │  ├─VL_4.교통정보.zip | 18 MB | 34127
            │  ├─VL_5.금융-은행.zip | 62 MB | 34128
            │  ├─VL_6.나이-생년월일.zip | 22 MB | 34129
            │  ├─VL_7.날짜-시간.zip | 33 MB | 34130
            │  ├─VL_8.단위.zip | 35 MB | 34131
            │  └─VL_9.사업자번호.zip | 8 MB | 34132
            ├─라벨링데이터_20230628_add
            │  └─VL_2.고유어기수_보완.zip | 148 MB | 246635
            └─원천데이터
                ├─VS_1.개인고유번호(음성).zip | 2 GB | 34133
                ├─VS_1.개인고유번호(텍스트).zip | 1 MB | 34134
                ├─VS_10.사이즈(음성).zip | 2 GB | 34135
                ├─VS_10.사이즈(텍스트).zip | 2 MB | 34136
                ├─VS_11.사투리(음성).zip | 155 MB | 34137
                ├─VS_11.사투리(텍스트).zip | 82 KB | 34138
                ├─VS_12.스포츠(음성).zip | 1 GB | 34139
                ├─VS_12.스포츠(텍스트).zip | 659 KB | 34140
                ├─VS_13.신분증번호(음성).zip | 6 GB | 34141
                ├─VS_13.신분증번호(텍스트).zip | 4 MB | 34142
                ├─VS_14.외래영어(음성).zip | 666 MB | 34143
                ├─VS_14.외래영어(텍스트).zip | 457 KB | 34144
                ├─VS_15.자동생성번호(음성).zip | 5 GB | 34145
                ├─VS_15.자동생성번호(텍스트).zip | 2 MB | 34146
                ├─VS_16.주문정보(음성).zip | 3 GB | 34147
                ├─VS_16.주문정보(텍스트).zip | 2 MB | 34148
                ├─VS_17.주소-구역(음성).zip | 7 GB | 34149
                ├─VS_17.주소-구역(텍스트).zip | 4 MB | 34150
                ├─VS_18.통계-수치(음성).zip | 12 GB | 34151
                ├─VS_18.통계-수치(텍스트).zip | 8 MB | 34152
                ├─VS_19.통신번호(음성).zip | 8 GB | 34153
                ├─VS_19.통신번호(텍스트).zip | 4 MB | 34154
                ├─VS_2.고유어기수(음성).zip | 16 GB | 34155
                ├─VS_2.고유어기수(텍스트).zip | 8 MB | 34156
                ├─VS_20.통화-금액(음성).zip | 10 GB | 34157
                ├─VS_20.통화-금액(텍스트).zip | 7 MB | 34158
                ├─VS_21.헬스케어(음성).zip | 3 GB | 34159
                ├─VS_21.헬스케어(텍스트).zip | 1 MB | 34160
                ├─VS_3.고유어서수(음성).zip | 701 MB | 34161
                ├─VS_3.고유어서수(텍스트).zip | 498 KB | 34162
                ├─VS_4.교통정보(음성).zip | 3 GB | 34163
                ├─VS_4.교통정보(텍스트).zip | 3 MB | 34164
                ├─VS_5.금융-은행(음성).zip | 12 GB | 34165
                ├─VS_5.금융-은행(텍스트).zip | 7 MB | 34166
                ├─VS_6.나이-생년월일(음성).zip | 4 GB | 34167
                ├─VS_6.나이-생년월일(텍스트).zip | 2 MB | 34168
                ├─VS_7.날짜-시간(음성).zip | 8 GB | 34169
                ├─VS_7.날짜-시간(텍스트).zip | 5 MB | 34170
                ├─VS_8.단위(음성).zip | 7 GB | 34171
                ├─VS_8.단위(텍스트).zip | 3 MB | 34172
                ├─VS_9.사업자번호(음성).zip | 2 GB | 34173
                └─VS_9.사업자번호(텍스트).zip | 856 KB | 34174"""

# ---------------------------------------- 이중 전사 template 코드 ----------------------------------------


def sentence_normalizer(sentence: str, side: str = "left") -> str:
    def normal_dual_transcript_extractor(script: str, select_side: Literal["left", "right"] = "left") -> str:
        """
        ETRI 전사규칙을 따른다면
            오른쪽: 철사
            왼쪽: 발음

        하지만 ETRI 전사 규칙을 따르지 않는 녀석들도 있어서 사용자가 정하도록 할 수 있도록 함.
        transcript_norm: Callable
        """
        normal_dual_bracket_regex = re.compile(r"\(([^()]+)\)/\(([^()]+)\)")

        # 비 정상적인 이중 전사 브라켓을 추출 함.
        bracket_iter = normal_dual_bracket_regex.finditer(script)
        select_side = 0 if select_side == "left" else 1

        diff = 0
        for bracket in bracket_iter:
            groups = bracket.groups()
            start_idx, end_idx = bracket.span()

            transcript_section = script[start_idx + diff : end_idx + diff]

            if not normal_dual_bracket_regex.search(transcript_section):
                raise ValueError(
                    f"이중 전사 구문을 추출하는 과정에서 값이 이상하게 바뀌었습니다.sentence: {transcript_section}"
                )

            extract_groups = groups[select_side]
            script = script[: start_idx + diff] + extract_groups + script[end_idx + diff :]
            diff = -(len(transcript_section)) + (len(extract_groups) + diff)

        return script

    # KsponSpeech 기준
    # 자/ 몸짱 열풍 다이어트에 성공하겠다.\xa0(5)/(오) 위 였구요.
    # 이런 애들 norm 할려고 remove_invisible_chars를 추가함.
    # sentence = remove_invisible_chars(sentence)
    sentence = normalize("NFC", sentence)
    sentence = normal_dual_transcript_extractor(sentence, side)
    sentence = sentence.strip()

    sentence = normalize("NFD", sentence)

    return sentence


# ---------------------------------------- 이중 전사 template 코드 ----------------------------------------


ds_logging.set_verbosity_info()
logger = ds_logging.get_logger("datasets")


@torch.inference_mode()
def apply_tnt_model(rank, world_size, dataset: dict, queue: Queue, max_num_gpus: int) -> None:
    setproctitle(f"classifier-{rank}")

    def collator(features):
        original_ls, spelling_ls, phonetic_ls, conversation_ls = [], [], [], []
        for item in features:
            if item is None:
                continue
            id_, feature = item
            conversations = [
                {"role": "spelling", "content": feature["scriptITN"]},
                {"role": "phonetic", "content": feature["scriptTN"].replace("[", "").replace("]", "")},
            ]
            text = pipe.tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=True,
            )
            spelling_ls.append(feature["scriptITN"])
            phonetic_ls.append(feature["scriptTN"].replace("[", "").replace("]", ""))
            conversation_ls.append(text)
            original_ls.append(feature)
        return (original_ls, spelling_ls, phonetic_ls, conversation_ls)

    device_id = rank % max_num_gpus

    chunker = zip_longest(*[iter(dataset.items())] * world_size, fillvalue=None)
    shard_script_ls = [chunk[rank] for chunk in chunker]
    shard_script_ls = [x for x in shard_script_ls if x is not None]  # None 제거

    pipe = pipeline(
        task="text2text-generation",
        model="jp1924/TNT0.5B-Dual",
        device=torch.device(f"cuda:{device_id}"),
        torch_dtype=torch.bfloat16,
    )
    pipe.model = torch.compile(pipe.model, mode="reduce-overhead")

    filter_ls, finish_ls = [], []
    data_loader = DataLoader(shard_script_ls, batch_size=4, collate_fn=collator, num_workers=0, pin_memory=False)
    p_bar = tqdm(total=len(shard_script_ls), desc=f"Rank {rank}", position=rank)
    for original_ls, spelling_ls, phonetic_ls, conversation_ls in data_loader:
        dual_script = pipe.tokenizer.batch_decode(
            [text["generated_token_ids"] for text in pipe(conversation_ls, return_tensors=True)]
        )
        cleaner = lambda txt: txt.replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|endoftext|>", "")  # noqa
        dual_script_ls = [cleaner(text.split("<|im_end|><|im_start|>")[-1]) for text in dual_script]

        norm_spelling_ls = [sentence_normalizer(spelling) for spelling in spelling_ls]
        norm_phonetic_ls = [sentence_normalizer(phonetic) for phonetic in phonetic_ls]

        norm_dual_spelling_ls = [sentence_normalizer(dual, side="left") for dual in dual_script_ls]
        norm_dual_phonetic_ls = [sentence_normalizer(dual, side="right") for dual in dual_script_ls]

        spelling_check_ls = [idx for idx, (x, y) in enumerate(zip(norm_spelling_ls, norm_dual_spelling_ls)) if x != y]
        phonetic_check_ls = [idx for idx, (x, y) in enumerate(zip(norm_phonetic_ls, norm_dual_phonetic_ls)) if x != y]
        filter_idx_ls = set(spelling_check_ls + phonetic_check_ls)

        # 정렬된 인덱스로 순서 보존
        filtered_idx = sorted(filter_idx_ls)
        kept_idx = [i for i in range(len(dual_script_ls)) if i not in filter_idx_ls]

        if filtered_idx:
            dual_filter_ls = [dual_script_ls[i] for i in filtered_idx]
            original_filter_ls = [original_ls[i] for i in filtered_idx]
            spelling_filter_ls = [norm_spelling_ls[i] for i in filtered_idx]
            phonetic_filter_ls = [norm_phonetic_ls[i] for i in filtered_idx]

            # 필터링 목록 저장
            for org, txt, spelling, phonetic in zip(
                original_filter_ls, dual_filter_ls, spelling_filter_ls, phonetic_filter_ls
            ):
                obj = {"id": org["scriptId"], "sentence": txt, "spelling": spelling, "phonetic": phonetic}
                filter_ls.append(obj)

        # 통과한(kept) 항목만 남김
        original_ls = [original_ls[i] for i in kept_idx]
        dual_script_ls = [dual_script_ls[i] for i in kept_idx]

        for org, txt in zip(original_ls, dual_script_ls):
            queue.put({"id": org["scriptId"], "sentence": txt})
            finish_ls.append(1)

        p_bar.set_description(f"finish: {len(finish_ls)} / {len(shard_script_ls)}, filter: {len(filter_ls)}")
        p_bar.update(len(conversation_ls))  # 배치 크기만큼 진행

    p_bar.close()


def pcm_to_wav_bytes(
    raw_pcm: bytes, *, sample_rate: int = SAMPLE_RATE, channels: int = 1, sampwidth: int = 2
) -> bytes:
    """
    raw PCM(헤더 없음)을 WAV 컨테이너로 감싸 bytes로 반환.
    - sample_rate: 샘플레이트(Hz)
    - channels: 채널 수(모노=1)
    - sampwidth: 샘플당 바이트 수(16-bit=2)
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_pcm)
    return buf.getvalue()


# https://github.com/huggingface/datasets/blob/dcd01046388fc052d37acc5a450bea69e3c57afc/templates/new_dataset_script.py#L65 참고해서 만듬.
class PatternedUtteranceWithNumber(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(name="Original", version="1.1.0", description="오리지널 데이터" + _DESCRIPTION),
        BuilderConfig(
            name="ASR", version="1.1.0", description="TNT모델을 활용해서 이중전사문으로 만든 데이터" + _DESCRIPTION
        ),
    ]

    DEFAULT_CONFIG_NAME = "ASR"
    DEFAULT_WRITER_BATCH_SIZE = 1000

    def _info(self) -> DatasetInfo:
        if self.config.name == "ASR":
            features = {
                "id": Value("string"),
                "audio": Audio(),
                "sentence": Value("string"),
                "metadata": {
                    "audio": {
                        "recordedID": Value("string"),
                        "recordedDate": Value("string"),
                        "recordedStart": Value("string"),
                        "fileName": Value("string"),
                        "filePath": Value("string"),
                        "recordedTime": Value("float32"),
                        "recordedQuality": Value("string"),
                        "recordedDevice": Value("string"),
                        "recordedDeviceName": Value("string"),
                        "recordedEnv": Value("string"),
                    },
                    "recordPerson": {
                        "collectedID": Value("string"),
                        "recorderID": Value("string"),
                        "sex": Value("string"),
                        "generation": Value("string"),
                        "residence": Value("string"),
                        "dialect": Value("string"),
                        "dialectRegion": Value("string"),
                        "areaInfomation": {"fatherHometown": Value("string"), "motherHometown": Value("string")},
                    },
                },
            }
        elif self.config.name == "Original":
            features = {
                "audio": {
                    "recordedID": Value("string"),
                    "recordedDate": Value("string"),
                    "recordedStart": Value("string"),
                    "fileName": Value("string"),
                    "filePath": Value("string"),
                    "recordedTime": Value("float32"),
                    "recordedQuality": Value("string"),
                    "recordedDevice": Value("string"),
                    "recordedDeviceName": Value("string"),
                    "recordedEnv": Value("string"),
                    "audio": Audio(),
                },
                "script": {
                    "scriptId": Value("string"),
                    "scriptITN": Value("string"),
                    "scriptTN": Value("string"),
                    "scriptNumberWord": Value("string"),
                    "patternTheme": Value("string"),
                },
                "recordPerson": {
                    "collectedID": Value("string"),
                    "recorderID": Value("string"),
                    "sex": Value("string"),
                    "generation": Value("string"),
                    "residence": Value("string"),
                    "dialect": Value("string"),
                    "dialectRegion": Value("string"),
                    "areaInfomation": {
                        "fatherHometown": Value("string"),
                        "motherHometown": Value("string"),
                    },
                },
            }

        return DatasetInfo(
            description=self.config.description,
            version=self.config.version,
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

        train_src_ls = [path for path in src_path_ls if "1.Training" in path.as_posix()]
        valid_src_ls = [path for path in src_path_ls if "2.Validation" in path.as_posix()]

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

    def _generate_examples(self, **kwagrs) -> Generator:
        if self.config.name == "ASR":
            for idx, data in enumerate(self._asr_generate_examples(**kwagrs)):
                yield idx, data
        elif self.config.name == "Original":
            for idx, data in enumerate(self._original_generate_examples(**kwagrs)):
                yield idx, data

    def _asr_generate_examples(self, file_ls: List[Path], split: str):
        def deploy_tnt_model(
            script_dataset: dict,
            world_size: int = 4,
            max_num_gpus: int = 4,
        ) -> List[dict]:
            max_num_gpus = torch.cuda.device_count()
            if max_num_gpus == 0:
                raise RuntimeError("No GPU available for inference.")

            manager = Manager()
            queue = manager.Queue()

            spawn(
                fn=apply_tnt_model,
                args=(world_size, script_dataset, queue, max_num_gpus),
                nprocs=world_size,
                join=True,
            )

            results = []
            for _ in range(queue.qsize()):
                results.append(queue.get())

            return {obj["id"]: obj["sentence"] for obj in results}

        src_path_ls = filter(lambda path: all(x in str(path) for x in ["원천데이터", "음성"]), file_ls)
        src_path_ls = natsorted(src_path_ls, key=lambda src_path: src_path.stem)

        lbl_path_ls = filter(lambda path: "라벨링데이터" in str(path), file_ls)
        lbl_path_ls = natsorted(lbl_path_ls, key=lambda lbl_path: lbl_path.stem)

        src_lbl_map = dict()
        for lbl_path in lbl_path_ls:
            lbl_id = lbl_path.stem.split(".")[0]
            src_ls = list()
            for src_path in src_path_ls:
                src_id = src_path.stem.split(".")[0]
                if lbl_id != src_id.replace("TS", "TL").replace("VS", "VL"):
                    continue
                src_ls.append(src_path)

            if not src_ls:
                continue
            src_lbl_map[lbl_id] = (lbl_path, src_ls)

        lbl_id_map = dict()
        for lbl_path in tqdm(lbl_path_ls, desc=f"{split}-id-script mapping"):
            lbl_zip = ZipFile(lbl_path, "r")
            lbl_info_ls = filter(lambda info: not info.is_dir(), lbl_zip.infolist())
            lbl_info_ls = natsorted(lbl_info_ls, key=lambda info: info.filename)

            # lbl_id = lbl_path.stem.split(".")[0]

            for lbl_info in lbl_info_ls:
                try:
                    labels = json.load(lbl_zip.open(lbl_info))
                except json.JSONDecodeError as e:
                    logger.warning(f"Label file {lbl_info.filename} is corrupted: {e}")
                    continue
                if labels["script"]["scriptId"] not in lbl_id_map:
                    lbl_id_map[labels["script"]["scriptId"]] = labels["script"]

        dual_transcript_map = deploy_tnt_model(lbl_id_map, world_size=2, max_num_gpus=2)

        for lbl_path, src_path_sub_ls in tqdm(list(src_lbl_map.values())):
            lbl_zip = ZipFile(lbl_path, "r")
            lbl_info_ls = filter(lambda info: not info.is_dir(), lbl_zip.infolist())
            lbl_info_ls = natsorted(lbl_info_ls, key=lambda info: info.filename)
            lbl_info_map = {Path(info.filename.split("_")[-1]).stem: info for info in lbl_info_ls}

            for src_path in src_path_sub_ls:
                src_zip = ZipFile(src_path, "r")
                src_info_ls = filter(lambda info: not info.is_dir(), src_zip.infolist())
                src_info_ls = natsorted(src_info_ls, key=lambda info: info.filename)

                for src_info in src_info_ls:
                    src_stem = Path(src_info.filename.split("_")[-1]).stem
                    if src_stem not in lbl_info_map:
                        logger.warning(f"Source file {src_stem} does not have a corresponding label file.")
                        continue

                    try:
                        lbl_info = lbl_info_map[src_stem]
                        label = json.load(lbl_zip.open(lbl_info))
                        if label["script"]["scriptId"] not in dual_transcript_map:
                            logger.warning(f"Label {label['script']['scriptId']} does not have a dual transcript.")
                            continue
                        audio = pcm_to_wav_bytes(src_zip.open(src_info).read())
                    except json.JSONDecodeError as e:
                        logger.warning(f"Label file {lbl_info.filename} is corrupted: {e}")
                        continue
                    except BaseException as e:
                        logger.warning(f"Error processing file {src_info.filename}: {e}")
                        continue

                    sentence = dual_transcript_map[label["script"]["scriptId"]]

                    yield {
                        "id": label["audio"]["recordedID"],
                        "audio": audio,
                        "sentence": sentence,
                        "metadata": {
                            "audio": label["audio"],
                            "recordPerson": label["recordPerson"],
                        },
                    }

    def _original_generate_examples(self, file_ls: List[Path], split: str):
        src_path_ls = filter(lambda path: all(x in str(path) for x in ["원천데이터", "음성"]), file_ls)
        src_path_ls = natsorted(src_path_ls, key=lambda src_path: src_path.stem)

        lbl_path_ls = filter(lambda path: "라벨링데이터" in str(path), file_ls)
        lbl_path_ls = natsorted(lbl_path_ls, key=lambda lbl_path: lbl_path.stem)

        src_lbl_map = dict()
        for lbl_path in lbl_path_ls:
            lbl_id = lbl_path.stem.split(".")[0]
            src_ls = list()
            for src_path in src_path_ls:
                src_id = src_path.stem.split(".")[0]
                if lbl_id != src_id.replace("TS", "TL").replace("VS", "VL"):
                    continue
                src_ls.append(src_path)
            if not src_ls:
                continue
            src_lbl_map[lbl_id] = (lbl_path, src_ls)
        for lbl_path, src_path_sub_ls in tqdm(list(src_lbl_map.values())):
            lbl_zip = ZipFile(lbl_path, "r")
            lbl_info_ls = filter(lambda info: not info.is_dir(), lbl_zip.infolist())
            lbl_info_ls = natsorted(lbl_info_ls, key=lambda info: info.filename)
            lbl_info_map = {Path(info.filename.split("_")[-1]).stem: info for info in lbl_info_ls}

            for src_path in src_path_sub_ls:
                src_zip = ZipFile(src_path, "r")
                src_info_ls = filter(lambda info: not info.is_dir(), src_zip.infolist())
                src_info_ls = natsorted(src_info_ls, key=lambda info: info.filename)

                for src_info in src_info_ls:
                    src_stem = Path(src_info.filename.split("_")[-1]).stem
                    if src_stem not in lbl_info_map:
                        logger.warning(f"Source file {src_stem} does not have a corresponding label file.")
                        continue
                    try:
                        lbl_info = lbl_info_map[src_stem]
                        label = json.load(lbl_zip.open(lbl_info))
                        audio = pcm_to_wav_bytes(src_zip.open(src_info).read())
                    except json.JSONDecodeError as e:
                        logger.warning(f"Label file {src_info.filename} is corrupted: {e}")
                        continue
                    except BaseException as e:
                        logger.warning(f"Error processing file {src_info.filename}: {e}")
                        continue

                    yield {
                        "audio": {
                            **label["audio"],
                            "audio": audio,
                        },
                        "script": label["script"],
                        "recordPerson": label["recordPerson"],
                    }
