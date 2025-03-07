import pandas as pd
from typing import Self
from pathlib import Path
from dspy import Example
from dataclasses import dataclass, field

from .utils import get_dir_name


@dataclass
class Dataset:
    source_path: Path
    model_name: str
    label_name: str = None
    input_names: list[str] = None
    name: str = field(init=False)
    answer_format: str = field(init=False)
    dataset: list[Example] = field(init=False)

    def __post_init__(self) -> None:
        self.name = get_dir_name(self.source_path)
        self.dataset = self._init_dataset()
        self.answer_format = self._init_ans_format()

    def __iter__(self) -> Example:
        for dspy_example in self.dataset:
            yield dspy_example

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, key) -> Self:
        self.dataset = self.dataset[key]
        return self

    def _init_dataset(self) -> list[Example]:
        data: pd.DataFrame = pd.read_csv(self.source_path)
        self.label_name, *inputs = data.columns
        self.input_names = inputs
        return [Example(**row).with_inputs(*inputs) for _, row in data.iterrows()]

    def _init_ans_format(self) -> str:
        formats = {
            "CommonsenseQA": "mc",
            "GSM8K": "number",
            "MultiArith": "number",
            "ObjectTracking": "text",
            "StrategyQA": "boolean",
        }
        return formats[self.name]

    def get_input_names(self, concat_str: str = ", ") -> str:
        return concat_str.join(self.input_names)
