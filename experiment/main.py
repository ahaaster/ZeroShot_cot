import dspy
import time
# from typing import Optional
from dataclasses import dataclass
from pathlib import Path

from utils import Num, load_dataset
from secret import secret


API_KEY = secret if secret else ""
LLM_MODEL = ["openai/gpt-4", "openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-3.5-turbo"]


# @dataclass
# class SignatureZeroShotCoT:
#     """Class for signature specification"""
#     context: str = None
#     question: str = "question"
#     intermediate: str = "reasoning"
#     answer: str = "answer"
#     answer_hint: str = ""

class ChoiceReasoning(dspy.Signature):
    """State the reasoning of arriving at a choice out of the given options"""
    query = dspy.InputField(prefix="q")
    choices = dspy.InputField(prefix="c", format="List[str]") 
    output = dspy.OutputField(desc="Let's think step by step.", prefix="a")

class ConclusionChoice(dspy.Signature):
    """We want you to choose one of the available choices"""
    query = dspy.InputField()
    output = dspy.OutputField(desc="Therefore, the answer is")



class Reasoning1(dspy.Signature):
    """"""
    query = dspy.InputField(prefix="q") 
    output = dspy.OutputField(desc="Let's think step by step.", prefix="a")

class ReasoningMisleading(dspy.Signature):
    """"""
    query = dspy.InputField() 
    output = dspy.OutputField(desc="By using the fact that the Earth is round")

class ReasoningIrrelevant(dspy.Signature):
    """"""
    query = dspy.InputField() 
    output = dspy.OutputField(desc="Bing bing wahooooo! Avada Kadavra")

class ConclusionSig(dspy.Signature):
    """We want a short answer without descriptive words"""
    query = dspy.InputField()
    output = dspy.OutputField(desc="Therefore, the answer is")


def main():
    chosen_model = LLM_MODEL[0]
    file_path = Path("dataset/zero-shot_cot/CommonsenseQA/data.csv")
    data = load_dataset(file_path=file_path)
    # print(data)
    
    lm = dspy.LM(chosen_model, max_tokens=2000, api_key=API_KEY)
    dspy.configure(lm=lm)
    
    # first_step = dspy.Predict(f"{sig.question} -> {sig.intermediate}")
    first_step = dspy.Predict(ChoiceReasoning)
    second_step = dspy.Predict(ConclusionChoice)
    # second_step = dspy.Predict(f"{sig.intermediate} -> {sig.answer}{sig.answer_hint}")
    
    for _, row in data.iterrows():
        question = row["question"]
        choices = row["choices"]
        label = row["labels"]

        reasoning = first_step(query=question, choices=choices)
        final_question = f"{question} \n\n{reasoning.output}"
        print(final_question)

        response = second_step(query=final_question)
        print(response.output)
        print(f"{label = }")
        
if __name__ == "__main__":
    main()
