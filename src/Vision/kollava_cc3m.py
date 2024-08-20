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
    Version,
)


URLS = {
    "image": "https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/resolve/main/images.zip",
    "label": "https://huggingface.co/datasets/tabtoyou/KoLLaVA-CC3M-Pretrain-595K/resolve/main/ko_chat.json",
}
_DESCRIPTION = """LLaVA에서 공개한 CC3M의 595K개 Visual Instruction dataset을 한국어로 번역한 데이터셋입니다. 기존 Ko-conceptual-captions에 공개된 한국어 caption을 가져와 데이터셋을 구축했습니다. 번역 결과가 다소 좋지 않아, 추후에 DeepL로 다시 번역할 수 있습니다."""
_VERSION = "1.0.0"


class KoLLaVAInsturct(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(name="chat", version=_VERSION, description=_DESCRIPTION),
    ]

    DEFAULT_CONFIG_NAME = "chat"

    def _info(self):
        features = Features(
            {
                "id": Value("string"),
                "image": Image(),
                "conversations": [{"role": Value("string"), "content": Value("string")}],
            }
        )
        return DatasetInfo(
            features=features,
            version=Version(_VERSION),
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
            img_split_regex = re.compile(rf"{img_token}|[^{img_token}]+")

            new_content_ls = list()
            for split_chat in img_split_regex.findall(content):
                new_content = {"type": "image"} if split_chat == img_token else {"type": "text", "text": split_chat}
                new_content_ls.append(new_content)

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
