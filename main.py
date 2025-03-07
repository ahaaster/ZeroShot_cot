import dspy
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from experiment import Dataset, prompt_control, basic_dspy
from experiment.utils import fetch_datasets, get_saved_data, get_dir_name
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
    "multihop": None,
}
PROMPT_METHOD = "basic"

METRICS = ["exact_match", "exact_lower", "semanticF1"]
METRIC = METRICS[0]


def main(experiment: str):
    if experiment == "prompt":
        method = PROMPT_METHOD
        chosen_model = LOCAL_MODEL
        chosen_datasets = Path("dataset/cot") / "CommonsenseQA"
        metric = METRIC
        run_prompts(method, chosen_model, chosen_datasets, metric, record_results=True)

    elif experiment == "metric_eval":
        evaluate_metrics()


def run_prompts(
    method: str,
    chosen_model: str,
    chosen_datasets: Path,
    metric_name: str,
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
            prompter(
                unrecorded,
                record_results=record_results,
                lm=lm,
                metric_name=metric_name,
            )

        # Edge case of something going very wrong
        elif n_unprompted != 0:
            raise ValueError(
                f"For some reason {n_unprompted = }, instead of a non-negative int"
            )

        # TODO: Evaluate the responses if no scores have been assigned


def create_scores_file():
    """This function should only be run once if the csv with recorded scores doesn't exist yet"""
    datasets = Path("dataset/cot").glob("**/*.csv")

    dataset_names = [get_dir_name(x) for x in sorted(datasets)]
    model_names = LOCAL_MODELS
    index = pd.MultiIndex.from_product(
        [dataset_names, model_names], names=["dataset", "model"]
    )

    method_names = [key for key in PROMPT_METHODS.keys()]
    metric_names = METRICS
    cols = pd.MultiIndex.from_product(
        [method_names, metric_names], names=["prompt_method", "metric"]
    )

    df = pd.DataFrame(np.nan, index=index, columns=cols)
    df.to_csv("results/scores.csv")


if __name__ == "__main__":
    # create_scores_file()
    main(EXPERIMENT)
