import json
import dspy
import time
from pathlib import Path
from secret import secret

API_KEY = secret if secret else ""
LLM_MODEL = ["openai/gpt-4", "openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-3.5-turbo"]

def load_dataset(file_path:[Path, str]=None):
    file_path = Path("dataset/zero-shot_cot/MultiArith/MultiArith.json") if file_path is None else file_path
    with open(file_path) as file:
        return json.load(file)

class ZeroShotReasoning(dspy.Signature):
    """
    """
    question: str = dspy.InputField()
    reasoning: str = dspy.OutputField()

class ZeroShotAnswer(dspy.Signature):
    question: str = dspy.InputField()
    answer: int = dspy.OutputField()

class ZeroShotAnswer2(dspy.Signature):
    reasoning: str = dspy.InputField()
    conclusion: int = dspy.OutputField()


def main():
    chosen_model = LLM_MODEL[0]
    data = load_dataset()
    
    lm = dspy.LM(chosen_model, max_tokens=2000, api_key=API_KEY)
    dspy.configure(lm=lm)

    frm = 150
    to = 151
    # module = dspy.Predict(f"question -> answer")
    step_reason = dspy.Predict(ZeroShotReasoning)
    step_answer = dspy.Predict(ZeroShotAnswer)
    for datum in data[frm:to]:
        question = datum['sQuestion']
        label = datum["lSolutions"][0]

        intermediate = step_reason(question=question)
        final_question = f"{question} \n\n{intermediate.reasoning}"
        
        prediction = step_answer(question=final_question)
        print(f"{final_question = } => {label = } \n{prediction.answer}\n")

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