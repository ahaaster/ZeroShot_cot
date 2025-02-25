import pandas as pd
from dspy import Example
from pathlib import Path
from dataclasses import dataclass, field

from .utils import get_dir_name


@dataclass
class Dataset:
    source_path: Path
    label_name: str = None
    input_names: list[str] = None
    name: str = field(init=False)
    dataset: list[Example] = field(init=False)

    def __post_init__(self):
        self.name = get_dir_name(self.source_path)
        self.dataset = self._init_dataset()

    def __iter__(self):
        for dspy_example in self.dataset:
            yield dspy_example

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        return self.dataset[key]

    def _init_dataset(self) -> list[Example]:
        data: pd.DataFrame = pd.read_csv(self.source_path)
        self.label_name, *inputs = data.columns
        self.input_names = inputs
        return [Example(**row).with_inputs(*inputs) for _, row in data.iterrows()]

    def get_input_names(self, concat_str: str = ", ") -> str:
        return concat_str.join(self.input_names)
