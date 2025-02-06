# import dspy
from json import load as read_json
from pandas import read_csv, DataFrame
from pathlib import Path
from typing import Any, Iterable
from dataclasses import dataclass, field
from dspy import LM, Example, Evaluate

LOCAL_MODELS = ["llama3.2:1b", "deepseek-r1:1.5b", "phi3.5", "gemma:2b", "qwen2.5:3b"]


@dataclass
class Dataset:
    source_path: Path
    label_name: str = None
    input_names: list[str] = None
    dataset_name: str = field(init=False)
    dataset: list[Example] = field(init=False)

    def __post_init__(self):
        self.dataset_name = get_dir_name(self.source_path)
        self.dataset = self._init_dataset()

    def __iter__(self):
        for example in self.dataset:
            yield example

    def __getitem__(self, key):
        return self.dataset[key]

    def _init_dataset(self) -> list[Example]:
        data: DataFrame = read_csv(self.source_path)
        self.label_name, *inputs = data.columns
        self.input_names = inputs
        return [Example(**row).with_inputs(*inputs) for _, row in data.iterrows()]

    def get_input_names(self, concat_str: str = ", ") -> str:
        return concat_str.join(self.input_names)

    def create_prompt(self, idx: int = 0) -> str:
        data = self.dataset[idx]
        prompt = [data[inpoet] for inpoet in self.input_names]
        return "\n".join(prompt)


def main():
    chosen_model = LOCAL_MODELS[0]
    lm = LM(
        f"ollama_chat/{chosen_model}",
        api_base="http://localhost:11434",
        api_key="",
        cache=False,
        temperature=0.9,
    )

    data_path = fetch_datasets(Path("cot/CommonsenseQA"))[0]
    dataset = Dataset(data_path)

    for idx, data in enumerate(dataset[20:24]):
        prompt = dataset.create_prompt(idx)
        print(prompt)
        print("-" * 30)
        resp1 = lm(messages=[{"role": "user", "content": prompt}])
        resp2 = lm(prompt)
        print(resp1)
        print(resp2)
        print("=" * 40)


def fetch_datasets(dir_path: Path | str = "") -> list[Path]:
    file_path = Path("dataset") / dir_path
    return sorted(file_path.glob("**/data.csv"))


def get_dir_name(file_path: Path) -> str:
    return file_path.parent.stem


if __name__ == "__main__":
    main()
