# import dspy
import pandas as pd
from pathlib import Path
from typing import Any, Iterable
from dataclasses import dataclass, field
from dspy import LM, Example, Evaluate
from tqdm import tqdm
import itertools

LOCAL_MODELS = ["llama3.2:1b", "deepseek-r1:1.5b", "phi3.5", "gemma:2b", "qwen2.5:3b"]
CHOSEN_MODEL = LOCAL_MODELS[0]


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


def main():
    chosen_model = CHOSEN_MODEL
    method = "base"

    lm = LM(
        f"ollama_chat/{chosen_model}",
        api_base="http://localhost:11434",
        api_key="",
        cache=False,
    )

    # query_path = Path("dataset/cot/CommonsenseQA")
    query_path = Path("dataset/cot")
    data_paths = fetch_datasets(query_path, file_name="data")

    for data_path in data_paths:
        dataset = Dataset(data_path)

        # Check if we already recorded some prompts
        results_dir = Path("results") / method / dataset.name
        model_name = convert_model_filename(chosen_model)
        results_path = fetch_datasets(results_dir, model_name)

        if results_path:
            prompts_recorded = len(pd.read_csv(results_path[0]))
            df_results = pd.read_csv(results_path[0])
        else:
            results_dir.mkdir(parents=True, exist_ok=True)
            prompts_recorded = 0
            df_results = pd.DataFrame()

        # Create batches of to be prompted queries
        amount_to_be_prompted: int = len(dataset) - prompts_recorded
        if amount_to_be_prompted == 0:
            continue

        unrecorded = dataset[-amount_to_be_prompted:]
        batch_n = 25
        batches: list[list[example]] = [
            unrecorded[i : i + batch_n] for i in range(0, len(unrecorded), batch_n)
        ]

        for data_batch in tqdm(batches):
            responses = []
            for example in tqdm(data_batch):
                prompt: str = create_prompt(example, dataset.input_names)
                response: list[str] = lm(messages=[{"role": "user", "content": prompt}])
                responses.append(
                    {
                        "prompt": prompt,
                        "response": response[0],
                        "ground_truth": example[dataset.label_name],
                    }
                )

            batch_df = pd.DataFrame(responses)
            df_results = pd.concat([df_results, batch_df])
            print(df_results.tail(3))
            df_results.to_csv(f"{results_dir}/{model_name}.csv", index=False)


def create_prompt(data: Example, input_keys: list[str], join_string: str = "\n") -> str:
    prompt = [data[key] for key in input_keys]
    return join_string.join(prompt)


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
