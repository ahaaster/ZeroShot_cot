"""
Prepare the dataset for the prompt framework implementation in this file.
Also construct the bespoke signature here that is used in the first phase of zero-shot CoT
"""
import json
import pandas as pd
from pathlib import Path

HERE = Path("dataset/zero-shot_cot/CommonsenseQA/")
RAW_SET = HERE / "dev_rand_split.jsonl"

def prepare_dataset():
    raw_df = pd.read_json(path_or_buf=RAW_SET, lines=True)
    nested_questions = raw_df["question"]

    df = pd.DataFrame(columns=["choices"])
    df["labels"] = raw_df["answerKey"]
    for idx, query in enumerate(nested_questions):
        df.at[idx, "question"] = query["stem"]
        
        choice_list = []
        choices = query["choices"]
        for choice in choices:
            key = choice["label"]
            value = choice["text"]
            choice = f"{key}) {value}"
            choice_list.append(choice)
        
        df.at[idx, "choices"] = choice_list
    
    return df


if __name__ == "__main__":
    result = prepare_dataset()
    df = pd.DataFrame(result)
    df.to_csv(HERE / "data.csv", index=False)

