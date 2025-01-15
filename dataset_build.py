import os
import traceback
from pathlib import Path

from datasets import load_dataset, load_dataset_builder
from setproctitle import setproctitle


def build_datasets(builder_path: Path, retry_count: int = 5) -> None:
    setproctitle(f"building-{builder_path.stem}")

    builder_class = load_dataset_builder(builder_path.as_posix(), trust_remote_code=True)
    for name in builder_class.builder_configs.keys():
        datasets = load_dataset(builder_path.as_posix(), name=name, trust_remote_code=True)
        for retries in range(retry_count):
            try:
                datasets.push_to_hub(
                    builder_path.stem,
                    private=True,
                    data_dir=name,
                    config_name=name,
                    max_shard_size="10GB",
                    token=os.getenv("HF_TOKEN", None),
                )
                break
            except BaseException as e:
                retry_count += 1
                traceback.print_exc()
                print(f"\n\n다음과 같은 애러가 발생해 재시작 함-{retries}: \n{e}\n\n")
        else:
            exit("재시도 횟수가 다 지나도 업로드가 불가능 했음. 프로그램을 종료함.")


if "__main__" in __name__:
    builder_path_ls = []

    for builder_path in builder_path_ls:
        if not isinstance(builder_path, Path):
            builder_path = Path(builder_path)

        build_datasets(builder_path)
