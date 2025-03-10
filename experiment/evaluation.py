import re
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field

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
            "number": r"-?\d*\,?\d+\.?\d*",
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
        if not string:
            return string

        elif self.answer_format == "mc":
            return re.sub(r"[\)|\.]", "", string)
        elif self.answer_format == "number":
            # Remove comma separator for magnitudes of 1,000
            string = re.sub(r",", "", string)
            return float(string)


@dataclass
class Metric(Module):
    name: str
    label_name: str = "label"
    output_name: str = "response"
    decoder: Decoder = None
    metric_func: callable = field(init=False)

    def __post_init__(self):
        metrics = {
            "exact_match": exact_match,
            "semanticF1": SemanticF1,
        }
        self.metric_func = metrics[self.name]

    def forward(self, example, pred, trace=None):
        label = example[self.label_name]
        resp = pred[self.output_name]

        if self.decoder is not None:
            resp = self.decoder(resp)
            resp = self.decoder.cleanup_string(resp)
        return self.metric_func(resp, label)


def exact_match(resp, label):
    if resp is None:
        print(
            f"HERE {resp = } | {type(resp)}, while it should be {label=} | {type(label)}"
        )
        return False

    elif isinstance(resp, int | float):
        return float(resp) == float(label)
    elif isinstance(resp, str):
        return str(resp).lower() == str(label).lower()


def evaluate_metrics() -> None:
    path_dataset = Path("dataset/cot/CommonsenseQA")
    label_path: list = fetch_datasets(path_dataset, file_name="data")
    df = pd.read_csv(label_path[0])
    label_series = df.loc[:6, "label"]

    shuffled_series = label_series.sample(frac=1)
    shuffled_series2 = "Therefore it is " + shuffled_series

    aligned_df = pd.concat([label_series, label_series], ignore_index=True, axis=1)
    # shuffled_df = pd.DataFrame()

    confirm_list = [
        "Therefore it is ",
        "Thus, ",
        "The most likely answer is ",
    ]

    for string in confirm_list:
        temp_series = string + label_series
        temp_df = pd.concat([label_series, temp_series], ignore_index=True, axis=1)
        aligned_df = pd.concat([aligned_df, temp_df], ignore_index=True)

    print(aligned_df.sample(frac=1))

    deny_list = [
        "Therefore it is not ",
        "Definitely not ",
        "I don't think it is ",
    ]

    nonsense_list = [
        "kjs hdjaksh djkahjkdhasjdhjjakha akhkasdj ",
        "me you, you me, mimimimimi ",
        "I don't want you to win for doing this, buddy ",
    ]

    df = pd.DataFrame([label_series.values, shuffled_series.values]).T

    # print(shuffled_series2)
