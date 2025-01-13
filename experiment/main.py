import json
import dspy
from pprint import pprint
from secret import sekrit_key
import time

LLM_MODEL = ["openai/gpt-4", "openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-3.5-turbo"]
lm = dspy.LM(LLM_MODEL[-1], max_tokens=2000, api_key=sekrit_key)
dspy.configure(lm=lm)

with open("dataset/MultiArith/MultiArith.json") as file:
    data = json.load(file)

list_hit = []
total_N = 0
frm = 100
to = 200
ans_format = ": int"
module = dspy.ChainOfThought(f"question -> answer{ans_format}")
for datum in data[frm:to]:
    question = datum['sQuestion']
    label = datum["lSolutions"][0]

    res = module(question=question)
    pred = res.get("answer")
    if label == pred:
        list_hit.append(pred)
    
    time.sleep(0.1)

print(f"Accuracy is: {len(list_hit)/len(data[frm:to]) * 100} %")
