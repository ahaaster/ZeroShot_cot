import dspy
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from experiment import Dataset, prompt_control, basic_dspy, dspy_prompt
from experiment.utils import fetch_datasets, get_saved_data, get_dir_name, save_score
from experiment.evaluation import evaluate_metrics, evaluate_results

EXPERIMENTS = ["prompt", "metric_eval"]
EXPERIMENT = EXPERIMENTS[0]

LOCAL_MODELS = ["llama3.2:1b", "deepseek-r1:1.5b", "phi3.5", "gemma:2b", "qwen2.5:3b"]
PROMPT_METHODS = {
    "control": prompt_control,
    "basic": basic_dspy,
    "constraint": dspy_prompt,
    "cot": dspy_prompt,
    "multihop": None,
}
METRICS = ["exact_match", "semanticF1", "semanticF1-decoded"]


def main(experiment: str):
    if experiment == "prompt":
        kwargs = {
            "method": "cot",
            "chosen_model": LOCAL_MODELS[2],
            "cache": False,
            "metric_name": METRICS[0],
            "record_results": True,
            "display_table": False,
            "greedy_first": False,
            "chosen_datasets": Path("dataset/cot") / "MultiArith",
        }
        kwargs["decode"] = False if kwargs["method"] == METRICS[1] else True
        run_prompts(**kwargs)

    elif experiment == "metric_eval":
        evaluate_metrics()


def run_prompts(
    chosen_datasets: Path,
    **kwargs,
):
    method = kwargs["method"]
    chosen_model = kwargs["chosen_model"]

    lm = dspy.LM(
        f"ollama_chat/{chosen_model}",
        api_base="http://localhost:11434",
        api_key="",
        cache=kwargs.pop("cache", True),
    )
    dspy.configure(lm=lm)
    prompter = PROMPT_METHODS[method]

    file_name = "data2" if method in ["cot", "constraint"] else "data"
    data_paths = fetch_datasets(chosen_datasets, file_name)

    for data_path in data_paths:
        dataset = Dataset(data_path, chosen_model, method)

        # Check if results are already recorded
        results_dir = Path("results") / method / dataset.name
        df_results = get_saved_data(results_dir, chosen_model)

        # Determine portion of dataset to be prompted
        n_unprompted: int = len(dataset) - len(df_results)

        if n_unprompted > 0:
            unrecorded: Dataset = dataset[-n_unprompted:]
            prompter(
                unrecorded,
                lm=lm,
                results_dir=results_dir,
                **kwargs,
            )

        # Edge case of something going very wrong
        elif n_unprompted != 0:
            raise ValueError(
                f"For some reason {n_unprompted = }, instead of a non-negative int"
            )

        # TODO: Evaluate the responses if no scores have been assigned
        else:
            score = evaluate_results(df_results, dataset.answer_format, **kwargs)
            score = round(score, 2)

            if not kwargs["record_results"]:
                print(f"{score = }")
                return

            save_score(
                score,
                model_name=chosen_model,
                method=method,
                metric_name=kwargs["metric_name"],
                dataset_name=dataset.name,
            )


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
