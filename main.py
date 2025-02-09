# import dspy
import pandas as pd
from pathlib import Path
from typing import Any, Iterable
from dataclasses import dataclass, field
from dspy import LM, Example, Evaluate
from tqdm import tqdm

LOCAL_MODELS = ["llama3.2:1b", "deepseek-r1:1.5b", "phi3.5", "gemma:2b", "qwen2.5:3b"]


def main():
    chosen_model = LOCAL_MODELS[-2]
    lm = LM(
        f"ollama_chat/{chosen_model}",
        api_base="http://localhost:11434",
        api_key="",
    )

    query_path = Path("dataset/cot/CommonsenseQA")
    data_paths = fetch_datasets(query_path, file_name="data")

    for data_path in data_paths:
        dataset = Dataset(data_path)
        responses = prompt_simple(dataset)
        record_responses(responses, "base", dataset.name, chosen_model)


def prompt_simple(dataset: Dataset) -> list[str]:
    responses = []
    for example in tqdm(dataset):
        prompt: str = create_prompt(example)
        resp: list = lm(messages=[{"role": "user", "content": prompt}])

        responses.append(
            {
                "prompt": prompt,
                "response": resp[0],
                "ground_truth": example[dataset.label_name],
            }
        )
    return responses


def record_responses(
    responses: list[str], method_name: str, dataset_name: str, chosen_model: str
) -> None:
    df = pd.DataFrame(responses)

    file_path = Path("results") / method_name / dataset_name
    file_path.mkdir(parents=True, exist_ok=True)
    model_name = convert_model_filename(chosen_model)
    df.to_csv(f"{file_path}/{model_name}.csv", index=False)


def create_prompt(data: Example, join_string: str = "\n") -> str:
    prompt = [data[key] for key in data._input_keys]
    return join_string.join(prompt)


@dataclass
class Dataset:
    source_path: Path
    label_name: str = None
    input_names: list[str] = None
    name: str = field(init=False)
    dataset: list[Example] = field(init=False)

    def __post_init__(self):
        self.name = get_dir_name(self.source_path)
        self.dataset = self._init_dataset()

    def __iter__(self):
        for example in self.dataset:
            yield example

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        return self.dataset[key]

    def _init_dataset(self) -> list[Example]:
        data: pd.DataFrame = pd.read_csv(self.source_path)
        self.label_name, *inputs = data.columns
        self.input_names = inputs
        return [Example(**row).with_inputs(*inputs) for _, row in data.iterrows()]

    def get_input_names(self, concat_str: str = ", ") -> str:
        return concat_str.join(self.input_names)


def fetch_datasets(dir_path: Path, file_name: str = "*") -> list[Path]:
    return sorted(dir_path.glob(f"**/{file_name}.csv"))


def get_dir_name(file_path: Path) -> str:
    return file_path.parent.stem


def convert_model_filename(model_name: str) -> str:
    return model_name.replace(".", "_").replace(":", "=")


def revert_model_filename(file_name: Path) -> str:
    file_name = file_name.stem
    return file_name.replace("_", ".").replace("=", ":")


if __name__ == "__main__":
    main()
