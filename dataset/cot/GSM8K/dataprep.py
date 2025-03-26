"""
Prepare the dataset for the prompt framework implementation in this file.
Also construct the bespoke signature here that is used in the first phase of zero-shot CoT
"""

import re
import json
import pandas as pd
from pathlib import Path

HERE = Path("dataset/cot/GSM8K")
RAW_SET = HERE / "test.jsonl"


def prepare_dataset():
    df = pd.read_json(path_or_buf=RAW_SET, lines=True)

    for idx, query in df.iterrows():
        label = query["answer"]
        label = label.split("\n#### ")[-1]
        label = re.sub(r",", "", label)
        df.at[idx, "answer"] = label

    df = df.rename(columns={"answer": "response"})
    return df[["response", "question"]]


if __name__ == "__main__":
    result = prepare_dataset()
    df = pd.DataFrame(result)
    df.iloc[:, 0] = df.iloc[:, 0].astype(int)
    df.to_csv(HERE / "data2.csv", index=False)
