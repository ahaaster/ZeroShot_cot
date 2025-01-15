import json
import pandas as pd
from pathlib import Path
from typing import Optional, Union

Num = Union[int, float]

def load_dataset(file_path:[Path, str]):
    file_path = Path("dataset/zero-shot_cot/MultiArith/MultiArith.json") if file_path is None else Path(file_path)
    if file_path.suffix == ".json":
        with open(file_path) as file:
            return json.load(file)
    elif file_path.suffix == ".csv":
        return pd.read_csv(file_path)
