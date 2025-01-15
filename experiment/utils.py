import json
from pathlib import Path
from typing import Optional, Union

Num = Union[int, float]

def load_dataset(file_path:[Path, str]):
    file_path = Path("dataset/zero-shot_cot/MultiArith/MultiArith.json") if file_path is None else file_path
    with open(file_path) as file:
        return json.load(file)
