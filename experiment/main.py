import json
import dspy
import time
from pathlib import Path
from secret import secret

API_KEY = secret if secret else ""

def load_dataset(file_path:[Path, str]=None):
    file_path = Path("dataset/zero-shot_cot/MultiArith/MultiArith.json") if file_path is None else file_path
    with open(file_path) as file:
        return json.load(file)

def main():
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

if __name__ == "__main__":
    main()
