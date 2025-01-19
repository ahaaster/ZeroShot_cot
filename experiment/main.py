import dspy
import time
import pandas as pd
from pathlib import Path
from dspy import Example, Evaluate, Predict
from dspy.evaluate import SemanticF1

from prompter import Reasoning, CoT
from evaluate import SemanticF1 as Similarity
from utils import load_dataset, get_directory_name, get_model_name
from secret import secret


LLM_MODEL = [
    "openai/gpt-3.5-turbo",
    "openai/gpt-4",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
]
METHODS = ["zero-shot", "cot_default", "cot_imitation", "zero-shot-cot"]
SCORING_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
RESULTS_PATH = Path("experiment/results.csv")

API_KEY = secret if secret else ""


def get_prompter(method_name, inputs, outputs="answer"):
    if isinstance(inputs, list):
        inputs = ", ".join(inputs)
    if isinstance(outputs, list):
        outputs = ", ".join(outputs)

    method_dict = {
        METHODS[0]: dspy.Predict(f"{inputs} -> {outputs}"),
        METHODS[1]: dspy.ChainOfThought(f"{inputs} -> {outputs}"),
        METHODS[2]: CoT(f"{inputs} -> {outputs}"),
        METHODS[-1]: Reasoning(
            inputs=inputs,
            outputs=outputs,
            # reasoning_hint="Avada Kadavra!"
        ),
    }
    return method_dict.get(method_name)


def main(
    chosen_model: str,
    method_name: str,
    file_path: Path,
    track_scores: bool = False,
    **kwargs,
) -> None:

    model_name = get_model_name(chosen_model)
    eval_config = kwargs.pop("config")

    data, dataset_name = load_dataset(file_path=file_path)
    label_name, *input_names = data.columns
    dataset = [Example(**row).with_inputs(*input_names) for _, row in data.iterrows()]

    lm = dspy.LM(chosen_model, max_tokens=2000, api_key=API_KEY)
    dspy.configure(lm=lm)

    prompter = get_prompter(method_name, input_names)

    for threshold in SCORING_THRESHOLDS[:1]:
        metric = Similarity(
            threshold=threshold,
            input_name=input_names[0],
            label_name=label_name,
            output_name=outputs,
        )

        evaluate = Evaluate(devset=dataset[:16], metric=metric, **eval_config)
        score = evaluate(prompter)

        if not track_scores:
            print(
                f"{model_name} had an accuracy of {score} % on the {dataset_name} dataset"
            )
            continue

        update_results(score, model_name, dataset_name, threshold, method)


def update_results(score, model_name, dataset_name, threshold_val, method):
    file_path = Path(f"experiment/{method}.csv")
    df = pd.read_csv(file_path, index_col=[0, 1])

    df.loc[(dataset_name, model_name), f"{threshold_val}"] = score
    df.to_csv(file_path)


# HELPER FUNCTION CONCERNING RECORDING SCORES
def create_results_file(method_name: str = None):
    """This function should only be run once if the csv with recorded results doesn't exist yet"""
    datasets = Path("dataset/zero-shot_cot").glob("**/*.csv")

    dataset_names = [get_directory_name(x) for x in sorted(datasets)]
    model_names = [model.split("/")[-1] for model in LLM_MODEL]
    iterables = [dataset_names, model_names]

    index = pd.MultiIndex.from_product(iterables, names=["dataset", "model"])
    cols = [str(x) for x in SCORING_THRESHOLDS]
    df = pd.DataFrame(0.0, index=index, columns=cols)

    file_path = Path(f"experiment/{method_name}.csv") if method_name else RESULTS_PATH
    df.to_csv(file_path)


# Copy pasted constants for easy visual access to set kwargs
#   LLM_MODEL = ["openai/gpt-3.5-turbo", "openai/gpt-4", "openai/gpt-4o-mini", "openai/gpt-4o"]
#   METHODS = ["zero-shot", "cot_default", "cot_imitation", "zero-shot-cot"]

if __name__ == "__main__":
    kwargs = {
        "chosen_model": LLM_MODEL[0],
        "method_name": METHODS[0],
        "track_scores": False,
        "config": {
            num_threads: 16,  # Adjust to available specs of local CPU for speed
            display_progress: True,
            display_table: 10,
            # provide_traceback: True,
            # max_error: 5,
            # return_all_scores: False,
            # return_outputs: False,
            # failure_score: 0.0,
        },
    }

    # RUNNING THIS FUNCTION WILL WIPE EXISTING FILE CLEAN
    # create_results_file(kwargs["method_name"])

    prepped_datasets = Path("dataset/zero-shot_cot").glob("**/data.csv")
    prepped_datasets: list[Path] = sorted(prepped_datasets)

    for file_path in prepped_datasets[3:4]:
        main(file_path=file_path, **kwargs)
