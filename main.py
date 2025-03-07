import dspy
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from experiment import Dataset, prompt_control, basic_dspy
from experiment.utils import fetch_datasets, get_saved_data
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
PROMPT_METHOD = "basic"


def main(experiment: str):
    if experiment == "prompt":
        method = PROMPT_METHOD
        chosen_model = LOCAL_MODEL
        chosen_datasets = Path("dataset/cot") / "CommonsenseQA"
        run_prompts(method, chosen_model, chosen_datasets, record_results=False)

    elif experiment == "metric_eval":
        evaluate_metrics()


def run_prompts(
    method: str,
    chosen_model: str,
    chosen_datasets: Path,
    *,
    record_results: bool = False,
):
    lm = dspy.LM(
        f"ollama_chat/{chosen_model}",
        api_base="http://localhost:11434",
        api_key="",
        cache=True,
    )
    dspy.configure(lm=lm)
    prompter = PROMPT_METHODS[method]

    data_paths = fetch_datasets(chosen_datasets, file_name="data")

    for data_path in data_paths:
        dataset = Dataset(data_path, chosen_model)

        # Check if results are already recorded
        results_dir = Path("results") / method / dataset.name
        df_results = get_saved_data(results_dir, chosen_model)

        # Determine portion of dataset to be prompted
        n_unprompted: int = len(dataset) - len(df_results)

        if n_unprompted > 0:
            unrecorded: Dataset = dataset[-n_unprompted:]
            print(method, unrecorded.name, f"{n_unprompted=}", type(unrecorded))
            prompter(unrecorded, chosen_model, record_results=record_results, lm=lm)
        # Edge case of something going very wrong
        elif n_unprompted != 0:
            raise ValueError(
                f"For some reason {n_unprompted = }, instead of a non-negative int"
            )

        # If results have been recorded without saved scores -> evaluate


if __name__ == "__main__":
    main(EXPERIMENT)
