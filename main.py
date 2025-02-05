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
    chosen_model = LOCAL_MODELS[1]
    lm = LM(
        f"ollama_chat/{chosen_model}",
        api_base="http://localhost:11434",
        api_key="",
        cache=False,
    )

    data_path = fetch_datasets(Path("cot/CommonsenseQA"))[0]
    dataset = Dataset(data_path)

    for data in dataset[:3]:
        print(data)

    # dataset = get_datasets(data_path)
    # labels, inputs = get_labels_and_inputs(dataset[0])
    # print(f"{labels= } | {inputs= }")
    # for data in dataset[:3]:
    #     print(f"{'='*30}\n{data[inputs[0]]}")
    #     print(data[labels[0]])
    #     prompt = f"{data[inputs[0]]}\n{data[inputs[1]]}"
    #     print(prompt)
    #     resp = lm(messages=[{"role": "user", "content": prompt}])
    #     print()
    #     print(resp)
    #     resp = lm(prompt)
    #     print(resp)


def get_labels_and_inputs(dataset: Example) -> (list[str], list[str]):
    # data = dataset[0] if isinstance(dataset, list) else dataset
    input_keys: set = data._input_keys
    keys: list[str] = data.keys()

    labels: list[str] = filter_list(keys, input_keys, include=False)
    inputs: list[str] = filter_list(keys, input_keys, include=True)
    return labels, inputs


def filter_list(
    item_list: Iterable, conditional: str | Iterable, include: bool = False
) -> list:
    if include:
        return [item for item in item_list if item in conditional]
    return [item for item in item_list if item not in conditional]


def get_datasets(data_path: Path = None) -> list[Example]:
    prepared_datasets: list[Path] = fetch_datasets(data_path)
    data, dataset_name = load_dataset(prepared_datasets[0])

    label_name, *input_names = data.columns
    return [Example(**row).with_inputs(*input_names) for _, row in data.iterrows()]


def fetch_datasets(dir_path: str | Path = "") -> list[Path]:
    file_path = Path("dataset") / dir_path
    return sorted(file_path.glob("**/data.csv"))


def load_dataset(file_path: Path) -> (Any, str):
    dir_name = get_dir_name(file_path)
    if file_path.suffix == ".json":
        with open(file_path) as file:
            return read_json(file), dir_name
    elif file_path.suffix == ".csv":
        return read_csv(file_path), dir_name


def get_dir_name(file_path: Path) -> str:
    return file_path.parent.stem


if __name__ == "__main__":
    main()
