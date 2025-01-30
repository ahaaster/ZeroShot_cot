# import dspy
from json import load as read_json
from pandas import read_csv
from pathlib import Path
from typing import Any, Iterable
from dspy import LM, Example, Evaluate

LOCAL_MODELS = ["llama3.2:1b", "deepseek-r1:1.5b", "phi3.5", "gemma:2b", "qwen2.5:3b"]


def main():
    chosen_model = LOCAL_MODELS[2]
    lm = LM(
        f"ollama_chat/{chosen_model}",
        api_base="http://localhost:11434",
        api_key="",
    )

    # x = lm("Hello World!", tempature=0.7)
    # y = lm(messages=[{"role": "user", "content": "Hello World!"}])
    # print(f"{x}\n\n{y}")

    data_path = Path("cot/CommonsenseQA")
    dataset = get_datasets(data_path)

    labels, inputs = get_labels_and_inputs(dataset)
    print(f"{labels= } | {inputs= }")
    for data in dataset[:3]:
        print(data[inputs[0]])
        resp = lm(
            messages=[
                {"role": "user", "content": f"{data[inputs[0]]}\n{data[inputs[1]]}"}
            ]
        )
        print(resp)


def get_labels_and_inputs(dataset: list[Example] | Example) -> (list[str], list[str]):
    data = dataset[0] if isinstance(dataset, list) else dataset
    input_keys: set = data._input_keys
    keys: list[str] = data.keys()

    labels = filter_list(keys, input_keys, include=False)
    inputs = filter_list(keys, input_keys, include=True)
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
