"""
Prepare the dataset for the prompt framework implementation in this file.
Also construct the bespoke signature here that is used in the first phase of zero-shot CoT
"""
import json
import pandas as pd
from pathlib import Path

HERE = Path("dataset/zero-shot_cot/GSM8K")
RAW_SET = HERE / "test.jsonl"

def prepare_dataset():
    df = pd.read_json(path_or_buf=RAW_SET, lines=True)
    
    for idx, query in df.iterrows():
        label = query["answer"]
        label = label.split("\n#### ")[-1]
        df.at[idx, "answer"] = label
        
    df = df.rename(columns={"answer": "label"})    
    return df[["label", "question"]]

if __name__ == "__main__":
    result = prepare_dataset()
    df = pd.DataFrame(result)
    df.to_csv(HERE / "data.csv", index=False)

