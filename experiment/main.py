import json
import dspy
import time
from pprint import pprint
from secret import secret

API_KEY = secret if secret else ""

LLM_MODEL = ["openai/gpt-4", "openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-3.5-turbo"]
lm = dspy.LM(LLM_MODEL[-1], max_tokens=2000, api_key=API_KEY)
dspy.configure(lm=lm)

with open("dataset/zero-shot_cot/MultiArith/MultiArith.json") as file:
    data = json.load(file)

list_hit = []
total_N = 0
frm = 150
to = 159
ans_format = ": int"
module = dspy.ChainOfThought(f"question -> answer{ans_format}")
for datum in data[frm:to]:
    question = datum['sQuestion']
    label = datum["lSolutions"][0]

    res = module(question=question)
    pred = res.answer
    if label == pred:
        list_hit.append(pred)
    
    time.sleep(0.2)
    print(f"done prompt #{len(list_hit)}")

print(f"Accuracy is: {len(list_hit)/len(data[frm:to]) * 100} %")
