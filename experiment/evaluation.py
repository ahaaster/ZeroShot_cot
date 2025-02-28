import re
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from dspy import Module
from dspy.evaluate import SemanticF1

from .dataset import Dataset
from .utils import fetch_datasets


@dataclass
class Decoder:
    answer_format: str = "text"
    greedy_first: bool = False  # True/False means respectively pick first/last

    def __post_init__(self):
        self.regex_format = self._init_regex()

    def __call__(self, string) -> str:
        return self.decode(string) if len(string) > 1 else string

    def _init_regex(self):
        regex_formats = {
            "text": r"([A-Z][^\.!?]*[\.!?])",  # Simply matches for full sentences
            "mc": r"[A-Z][\)|\.]",  # Multiple Choice
            "number": r"-?\d+\.?\d*",
            "boolean": r"([tT]rue|[fF]alse|[Uu]ntrue|[yY]es|[nN]o\b|\w*[Nn].t\s\w*\s?true)",
        }
        return regex_formats[self.answer_format]

    def decode(self, string: str) -> str:
        matches = re.findall(self.regex_format, string)

        if not matches:
            return ""
        elif self.greedy_first:
            return matches[0]
        else:
            return matches[-1]

    def cleanup_string(self, string: str) -> str:
        if self.answer_format == "mc":
            return re.sub(r"[\)|\.]", "", string)


@dataclass
class Metric(Module):
    metric_func: callable
    label_name: str = "label"
    output_name: str = "response"
    decoder: Decoder = None

    def forward(self, example, pred, trace=None):
        label = example[self.label_name]
        resp = pred[self.output_name]

        if self.decoder is not None:
            resp = self.decoder(resp)
            resp = self.decoder.cleanup_string(resp)
        return self.metric_func(resp, label)


def exact_match(resp, label):
    return resp == label


def exact_match_lower(resp: str, label: str) -> bool:
    return resp.lower() == label.lower()


def evaluate_metrics() -> None:
    path_dataset = Path("dataset/cot/MultiArith")
    label_path: list = fetch_datasets(path_dataset, file_name="data")
    df = pd.read_csv(label_path[0])
    label_series = df.loc[:, "label"]

    shuffled_series = label_series.sample(frac=1)
    print(shuffled_series.head())
