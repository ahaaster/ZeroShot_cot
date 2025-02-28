import dspy
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from experiment import Dataset, prompt_control, basic_dspy
from experiment.utils import fetch_datasets
from experiment.evaluation import evaluate_metrics

EXPERIMENTS = ["prompt", "metric_eval"]
EXPERIMENT = EXPERIMENTS[0]

# Prompt 'settings'
LOCAL_MODELS = ["llama3.2:1b", "deepseek-r1:1.5b", "phi3.5", "gemma:2b", "qwen2.5:3b"]
LOCAL_MODEL = LOCAL_MODELS[1]

PROMPT_METHODS = {
    "control": prompt_control,
    "basic": basic_dspy,
    "constraint": None,
    "cot": None,
    "multihop?": None,
}
PROMPT_METHOD = PROMPT_METHODS["basic"]


def main(experiment: str):
    if experiment == "prompt":
        method = PROMPT_METHOD
        chosen_model = LOCAL_MODEL
        chosen_datasets = Path("dataset/cot") / "CommonsenseQA"
        run_prompts(method, chosen_model, chosen_datasets, record_results=False)

    elif experiment == "metric_eval":
        evaluate_metrics()


def run_prompts(
    prompter: callable,
    chosen_model: str,
    chosen_datasets: Path,
    *,
    record_results: bool = False,
):
    lm = dspy.LM(
        f"ollama_chat/{chosen_model}",
        api_base="http://localhost:11434",
        api_key="",
        # cache=False,
    )
    dspy.configure(lm=lm)

    data_paths = fetch_datasets(chosen_datasets, file_name="data")

    for data_path in data_paths:
        dataset = Dataset(data_path)
        prompter(dataset, chosen_model, record_results=record_results, lm=lm)


if __name__ == "__main__":
    main(EXPERIMENT)
