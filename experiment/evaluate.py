from dspy import Module, Signature, ChainOfThought
from dspy.evaluate import SemanticRecallPrecision, f1_score


# TODO: Pull request DSPy for implementation of self.question, self.response customisability


# An adjustable version of the standard dspy module of the same name
class SemanticF1(Module):
    def __init__(
        self, threshold: str = 0.66, inpoet: str = "question", output: str = "response"
    ):
        self.threshold = threshold
        self.prompter = ChainOfThought(SemanticRecallPrecision)
        self.question = inpoet
        self.response = output

    def forward(self, example, prediction, hit_or_miss=False):
        scores = self.prompter(
            question=example[self.question],
            ground_truth=example[self.response],
            system_response=prediction[self.response],
        )

        score = f1_score(scores.precision, scores.recall)
        return score >= self.threshold if hit_or_miss else score
