import pandas as pd
from typing import Self
from pathlib import Path
from dataclasses import dataclass, field

from dspy import Example, Module, Signature
from dspy import Predict, ChainOfThought

from .utils import get_dir_name


@dataclass
class Dataset:
    source_path: Path
    model_name: str
    method_name: str
    label_name: str = None
    input_names: list[str] = None
    name: str = field(init=False)
    answer_format: str = field(init=False)
    dataset: list[Example] = field(init=False)
    constraint: str = field(init=False)
    prompter: Module = field(init=False)

    def __post_init__(self) -> None:
        self.name = get_dir_name(self.source_path)
        self.dataset = self._init_dataset()
        self.answer_format = self._init_ans_format()
        self.constraint = self._init_constraint()
        self.prompter = ChainOfThought if self.method_name == "cot" else Predict

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

    def _init_constraint(self):
        # constrained = True if self.method_name in ["constraint"] else False
        constrained = True if self.method_name in ["constraint", "cot"] else False

        if not constrained:
            return None

        constraints = {
            "mc": None,
            "text": None,
            "number": "float",
            "boolean": "bool",
        }
        return constraints[self.answer_format]

    def get_input_names(self, concat_str: str = ", ") -> str:
        return concat_str.join(self.input_names)

    def create_signature(self, output_name: str = None) -> Signature:
        inputs = self.get_input_names()
        output = output_name or self.label_name
        constraint = self.constraint

        if constraint:
            output += f": {constraint}"

        return Signature(f"{inputs} -> {output}")

    def create_prompt_kwargs(self, example: Example) -> dict:
        kwargs = {}
        for inpt in self.input_names:
            kwargs[inpt] = example[inpt]

        return kwargs
