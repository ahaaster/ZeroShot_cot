import dspy
import pandas as pd
from pathlib import Path
from dspy.evaluate import SemanticF1

from .dataset import Dataset
from .utils import fetch_datasets


def exact_match(resp: list[str], ground_truth: list[str]) -> float:
    pass


def decoding_score(resp: list[str], ground_truth: list[str]) -> float:
    pass


def semantic_scoreing(resp: list[str], ground_truth: list[str]) -> float:
    pass


def evaluate_metrics():
    path_dataset = Path("dataset/cot/MultiArith")
    label_path: list = fetch_datasets(path_dataset, file_name="data")
    df = pd.read_csv(label_path[0])
    label_series = df.loc[:, "label"]

    shuffled_series = label_series.sample(frac=1)
    print(shuffled_series.head())
