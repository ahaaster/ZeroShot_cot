"""
Prepare the dataset for the prompt framework implementation in this file.
Also construct the bespoke signature here that is used in the first phase of zero-shot CoT
"""
import json
import pandas as pd
from pathlib import Path

HERE = Path("dataset/zero-shot_cot/MultiArith/")
RAW_SET = HERE / "MultiArith.json"

def prepare_dataset():
    with open(RAW_SET) as file:
        dataset = json.load(file)
    
    result = []
    for data in dataset:
        result.append({
            "question": data["sQuestion"],
            "label": int(data["lSolutions"][0])
        })
    
    return result    

if __name__ == "__main__":
    result = prepare_dataset()
    df = pd.DataFrame(result)
    df.to_csv(HERE / "data.csv", index=False)

