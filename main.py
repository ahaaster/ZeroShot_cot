import dspy
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from experiment import Dataset, prompt_control, fetch_datasets

LOCAL_MODELS = ["llama3.2:1b", "deepseek-r1:1.5b", "phi3.5", "gemma:2b", "qwen2.5:3b"]
PROMPT_METHODS = {
    "control": prompt_control,
    "basic": None,
    "constraint": None,
    "zeroshot-cot": None,
    "multihop?": None,
}


def main():
    method = "control"
    chosen_model = LOCAL_MODELS[0]
    chosen_datasets = Path("dataset/cot")
    prompter = PROMPT_METHODS[method]

    lm = dspy.LM(
        f"ollama_chat/{chosen_model}",
        api_base="http://localhost:11434",
        api_key="",
        cache=False,
    )

    data_paths = fetch_datasets(chosen_datasets, file_name="data")

    for data_path in data_paths:
        dataset = Dataset(data_path)
        prompter(lm, dataset, chosen_model)


if __name__ == "__main__":
    main()
