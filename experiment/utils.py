import json
import pandas as pd
from pathlib import Path
from typing import Any


def load_dataset(file_path: Path) -> (Any, str):
    dir_name = get_directory_name(file_path)
    if file_path.suffix == ".json":
        with open(file_path) as file:
            return json.load(file), dir_name
    elif file_path.suffix == ".csv":
        return pd.read_csv(file_path), dir_name


def get_directory_name(file_path: Path) -> str:
    return file_path.parent.stem


def get_model_name(model: str) -> str:
    return model.split("/")[-1]


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
