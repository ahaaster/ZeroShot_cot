import dspy
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from experiment import Dataset, prompt_control
from experiment.utils import fetch_datasets
from experiment.evaluation import evaluate_metrics

EXPERIMENTS = ["prompt", "metric_eval"]
EXPERIMENT = EXPERIMENTS[1]

# Prompt 'settings'
LOCAL_MODELS = ["llama3.2:1b", "deepseek-r1:1.5b", "phi3.5", "gemma:2b", "qwen2.5:3b"]
PROMPT_METHODS = {
    "control": prompt_control,
    "basic": None,
    "constraint": None,
    "cot": None,
    "multihop?": None,
}


def main(experiment: str):
    if experiment == "prompt":
        method = "control"
        chosen_model = LOCAL_MODELS[1]
        chosen_datasets = Path("dataset/cot")
        run_prompts(method, chosen_model, chosen_datasets, record_results=False)

    elif experiment == "metric_eval":
        evaluate_metrics()


def run_prompts(
    method: str, chosen_model: str, chosen_datasets: Path, record_results: bool = False
):
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
        prompter(lm, dataset, chosen_model, record_results)


if __name__ == "__main__":
    main(EXPERIMENT)
