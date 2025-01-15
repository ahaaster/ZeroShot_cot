import dspy
import time
# from typing import Optional
from dataclasses import dataclass

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
    data = load_dataset(file_path=None)
    
    lm = dspy.LM(chosen_model, max_tokens=2000, api_key=API_KEY)
    dspy.configure(lm=lm)
    
    # first_step = dspy.Predict(f"{sig.question} -> {sig.intermediate}")
    first_step = dspy.Predict(Reasoning1)
    second_step = dspy.Predict(ConclusionSig)
    # second_step = dspy.Predict(f"{sig.intermediate} -> {sig.answer}{sig.answer_hint}")
    
    for datum in data[210:211]:
        question = datum['sQuestion']
        label = datum["lSolutions"][0]

        reasoning = first_step(query=question)
        final_question = f"{question} \n\n{reasoning.output}"
        print(final_question)

        response = second_step(query=final_question)
        print(response.output)
        print(f"{label = }")
        
if __name__ == "__main__":
    main()


""" OLD CODE
data = load_dataset()
    
    LLM_MODEL = ["openai/gpt-4", "openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-3.5-turbo"]
    lm = dspy.LM(LLM_MODEL[0], max_tokens=2000, api_key=API_KEY)
    dspy.configure(lm=lm)

    list_hit = []
    total_N = 0
    frm = 150
    to = 152
    # ans_format = ": int"
    # module = dspy.ChainOfThought(f"question -> answer{ans_format}")
    module = dspy.Predict(f"question -> answer: int")
    for datum in data[frm:to]:
        question = datum['sQuestion']
        label = datum["lSolutions"][0]

        # questions = [question, question, question]

        res = module(question=question)
        print(f"{question = } \n{label = } \n{res}")
        # pred = res.answer
        # if label == pred:
        #     list_hit.append(pred)
        
        # time.sleep(0.2)
        # print(f"done prompt #{len(list_hit)}")

    # print(f"Accuracy is: {len(list_hit)/len(data[frm:to]) * 100} %")
"""