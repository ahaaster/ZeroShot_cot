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
