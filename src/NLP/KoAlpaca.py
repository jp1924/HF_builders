from typing import Generator, List

from datasets import (
    BuilderConfig,
    DatasetInfo,
    Features,
    GeneratorBasedBuilder,
    Split,
    SplitGenerator,
    Value,
    load_dataset,
)


_LICENSE = """Apache License 2.0"""

_CITATION = r"""@misc{alpaca,
  author = {Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {Stanford Alpaca: An Instruction-following LLaMA model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/stanford_alpaca}},
}"""

_DESCRIPTION = """KoAlpaca-v1.1a\nhttps://github.com/Beomi/KoAlpaca"""
_VERSION = "1.1.0"
_HOMEPAGE = "https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a"


class KoAlpaca(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(name="SFT", version=_VERSION, description=_DESCRIPTION),
    ]
    DEFAULT_CONFIG_NAME = "SFT"

    def _info(self) -> DatasetInfo:
        self.features = Features(
            {
                "id": Value("string"),
                "conversations": [{"role": Value("string"), "content": Value("string")}],
                "prompt": Value("string"),
                "answer": Value("string"),
            }
        )

        return DatasetInfo(
            description=_DESCRIPTION,
            features=self.features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            version=_VERSION,
        )

    def _split_generators(self, _) -> List[SplitGenerator]:
        dataset = load_dataset("beomi/KoAlpaca-v1.1a")

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "dataset": dataset["train"],
                },
            ),
        ]

    def _generate_examples(self, dataset) -> Generator:
        for idx, data in enumerate(dataset):
            conversations = [
                {"role": "user", "content": data["instruction"]},
                {"role": "assistant", "content": data["output"]},
            ]
            _id = data["url"].replace("https://kin.naver.com/qna/detail.naver?", "")
            return (
                idx,
                {
                    "id": _id,
                    "conversations": conversations,
                    "prompt": data["instruction"],
                    "answer": data["output"],
                },
            )
