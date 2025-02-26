import re
import dspy
import pandas as pd
from pathlib import Path
from dspy.evaluate import SemanticF1

from .dataset import Dataset
from .utils import fetch_datasets


def exact_match(resp: str, label: str) -> float:
    return resp == label


def decode_response(response: str, answer_type: str, last: bool = False) -> str:
    regex_formats = {
        "number": r"-?\d+\.?\d*",
        "multiple choice": r"[A-Z][\)|\.]",
        "boolean": r"([tT]rue|[fF]alse|untrue|[yY]es|[nN]o\b|\w*[Nn].t\s\w*\s?true)",
        "text": r"([A-Z][^\.!?]*[\.!?])",  # Simply matches for full sentences
    }

    regex_string = regex_formats[answer_type]
    matches = re.findall(regex_string, response)

    if not matches:
        return ""
    elif last:
        return matches[-1]
    else:
        return matches[0]


def decode_match(
    response: str, label: str, answer_type: str, last: bool = False
) -> str:
    decoded_resp = decode_response(response, answer_type, last)
    return exact_match(decoded_resp, label)


def evaluate_metrics():
    path_dataset = Path("dataset/cot/MultiArith")
    label_path: list = fetch_datasets(path_dataset, file_name="data")
    df = pd.read_csv(label_path[0])
    label_series = df.loc[:, "label"]

    shuffled_series = label_series.sample(frac=1)
    print(shuffled_series.head())
