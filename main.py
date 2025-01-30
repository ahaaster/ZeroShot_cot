import dspy
import pandas as pd
from pathlib import Path
from typing import Any
from dspy import Example

LOCAL_MODELS = ["llama3.2:1b", "deepseek-r1:1.5b", "phi3.5", "gemma:2b", "qwen2.5:3b"]


def main():
    chosen_model = LOCAL_MODELS[2]
    lm = dspy.LM(
        f"ollama_chat/{chosen_model}",
        api_base="http://localhost:11434",
        api_key="",
        # cache=False,
    )

    x = lm("Hello World!", tempature=0.7)
    y = lm(messages=[{"role": "user", "content": "Hello World!"}])
    print(f"{x}\n\n{y}")

    data_path = Path("cot/MultiArith")
    data = get_datasets(data_path)
    print(data[:2])


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
            return json.load(file), dir_name
    elif file_path.suffix == ".csv":
        return pd.read_csv(file_path), dir_name


def get_dir_name(file_path: Path) -> str:
    return file_path.parent.stem


if __name__ == "__main__":
    main()
