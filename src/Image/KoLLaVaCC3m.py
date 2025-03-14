import json
import re
from pathlib import Path

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


URLS = {
    "image": "https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/resolve/main/images.zip",
    "label": "https://huggingface.co/datasets/tabtoyou/KoLLaVA-CC3M-Pretrain-595K/resolve/main/ko_chat.json",
}

_HOMEPAGE = "https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K"


_DATANAME = "KoLLaVaCC3m"


_DESCRIPTION = """LLaVA에서 공개한 CC3M의 595K개 Visual Instruction dataset을 한국어로 번역한 데이터셋입니다. 기존 Ko-conceptual-captions에 공개된 한국어 caption을 가져와 데이터셋을 구축했습니다. 번역 결과가 다소 좋지 않아, 추후에 DeepL로 다시 번역할 수 있습니다."""


class KoLLaVaCC3m(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(name="SFT", version="1.0.0", description=_DESCRIPTION),
    ]

    DEFAULT_CONFIG_NAME = "SFT"
    DEFAULT_WRITER_BATCH_SIZE = 1000

    def _info(self):
        features = Features(
            {
                "id": Value("string"),
                "image": Image(),
                "conversations": [{"role": Value("string"), "content": Value("string")}],
            }
        )
        return DatasetInfo(
            description=self.config.description,
            version=self.config.version,
            features=features,
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(URLS)

        image = downloaded_files.pop("image")
        image_file_ls = [x for x in Path(image).rglob("*") if x.is_file()]
        image_file_table = {x.name: x for x in image_file_ls}

        label = downloaded_files.pop("label")
        label = json.loads(Path(label).read_text())

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "label_ls": label,
                    "image_dict": image_file_table,
                },
            ),
        ]

    def _generate_examples(self, label_ls, image_dict):
        def convert_mm_content(content: str, img_token: str):
            img_split_regex = re.compile(rf"{img_token}|.")

            new_content_ls = list()
            sentence = ""
            for token in img_split_regex.findall(content):
                if token == img_token:
                    if sentence:
                        new_content_ls.append({"type": "text", "text": sentence})
                        sentence = ""
                    new_content_ls.append({"type": "image"})
                    continue

                sentence += token
            else:
                if sentence:
                    new_content_ls.append({"type": "text", "text": sentence})

            return new_content_ls

        for idx, data in enumerate(label_ls):
            if data["image"] not in image_dict:
                continue

            new_conversations_ls = list()
            for chat in data["conversations"]:
                new_conversations_ls.append(
                    {
                        "role": "user" if chat["from"] == "human" else "assistant",
                        "content": json.dumps(convert_mm_content(chat["value"], "<image>"), ensure_ascii=False),
                    }
                )

            data = {
                "id": data["id"],
                "image": image_dict[data["image"]].read_bytes(),
                "conversations": new_conversations_ls,
            }

            yield (idx, data)
