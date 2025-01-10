from datasets import (
    BuilderConfig,
    Dataset,
    DatasetInfo,
    Features,
    GeneratorBasedBuilder,
    Split,
    SplitGenerator,
    Value,
    Version,
    load_dataset,
)


class KoVariousDataset(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="SFT",
            data_dir="SFT",
            version=Version("1.0.0"),
            description="kyujinpy/Ko-various-dataset를 conversations로 바꾼 데이터,",
        ),
    ]
    DEFAULT_CONFIG_NAME = "SFT"

    def _info(self):
        features = Features(
            {
                "id": Value("string"),
                "conversations": [{"role": Value("string"), "content": Value("string")}],
                "system": Value("string"),
                "prompt": Value("string"),
                "answer": Value("string"),
            }
        )

        return DatasetInfo(
            features=features,
            version=self.config.version,
            description=self.config.version,
        )

    def _split_generators(self, _):
        dataset = load_dataset("kyujinpy/Ko-various-dataset", split="train")
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "dataset": dataset,
                },
            ),
        ]

    def _generate_examples(self, dataset: Dataset):
        for idx, data in enumerate(dataset):
            conversations = [
                {"role": "user", "content": data["instruction"]},
                {"role": "assistant", "content": data["output"]},
            ]
            yield (
                idx,
                {
                    "id": data["id"],
                    "conversations": conversations,
                    "system": data["input"],
                    "prompt": data["instruction"],
                    "answer": data["output"],
                },
            )
