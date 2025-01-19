import dspy
import pandas as pd
from pathlib import Path
from dspy import Example, Evaluate, Predict
from dspy.evaluate import SemanticF1

from prompter import select_prompter
from evaluate import SemanticF1 as Similarity
from utils import get_directory_name, get_model_name
from utils import create_results_file, load_dataset
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


def main(
    chosen_model: str,
    method_name: str,
    file_path: Path,
    track_scores: bool = False,
    **kwargs,
) -> None:

    model_name = get_model_name(chosen_model)
    lm_config = kwargs.pop("lm_config")
    eval_config = kwargs.pop("eval_config")

    data, dataset_name = load_dataset(file_path=file_path)
    label_name, *input_names = data.columns
    dataset = [Example(**row).with_inputs(*input_names) for _, row in data.iterrows()]

    lm = dspy.LM(chosen_model, **lm_config)
    dspy.configure(lm=lm)

    prompter = select_prompter(method_name, input_names)

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


# Copy pasted constants for easy visual access to set kwargs
#   LLM_MODEL = ["openai/gpt-3.5-turbo", "openai/gpt-4", "openai/gpt-4o-mini", "openai/gpt-4o"]
#   METHODS = ["zero-shot", "cot_default", "cot_imitation", "zero-shot-cot"]

if __name__ == "__main__":
    kwargs = {
        "chosen_model": LLM_MODEL[0],
        "method_name": METHODS[0],
        "track_scores": False,
        "lm_config": {"max_tokens": 2000, "api_key": API_KEY},
        "eval_config": {
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
