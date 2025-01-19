from dspy import Module, Signature, ChainOfThought
from dspy.evaluate import (
    SemanticRecallPrecision,
    f1_score,
    DecompositionalSemanticRecallPrecision,
)


# TODO: Pull request DSPy for implementation of self.question, self.response customisability


# An adjustable version of the standard dspy module of the same name
class SemanticF1(Module):
    def __init__(
        self,
        threshold: str = 0.66,
        input_name: str = "question",
        label_name: str = "response",
        output_name: str = "response",
    ):
        self.threshold = threshold
        self.prompter = ChainOfThought(SemanticRecallPrecision)
        self.question = input_name
        self.example_resp = label_name
        self.response = output_name

    def forward(self, example, pred, trace=None):
        scores = self.prompter(
            question=example[self.question],
            ground_truth=example[self.example_resp],
            system_response=pred[self.response],
        )

        score = f1_score(scores.precision, scores.recall)
        return score if trace is None else score >= self.threshold
