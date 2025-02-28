import re
import dspy
import pandas as pd
from pathlib import Path
from dspy.evaluate import SemanticF1

from .dataset import Dataset
from .utils import fetch_datasets


def exact_match(resp: str, label: str) -> bool:
    return resp.lower() == label.lower()


def decode_response(response: str, answer_type: str, last: bool = False) -> str:
    regex_formats = {
        "text": r"([A-Z][^\.!?]*[\.!?])",  # Simply matches for full sentences
        "multiple choice": r"[A-Z][\)|\.]",
        "number": r"-?\d+\.?\d*",
        "boolean": r"([tT]rue|[fF]alse|[Uu]ntrue|[yY]es|[nN]o\b|\w*[Nn].t\s\w*\s?true)",
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
) -> bool:
    decoded_resp = decode_response(response, answer_type, last)
    return exact_match(decoded_resp, label)


def evaluate_metrics() -> None:
    path_dataset = Path("dataset/cot/MultiArith")
    label_path: list = fetch_datasets(path_dataset, file_name="data")
    df = pd.read_csv(label_path[0])
    label_series = df.loc[:, "label"]

    shuffled_series = label_series.sample(frac=1)
    print(shuffled_series.head())
