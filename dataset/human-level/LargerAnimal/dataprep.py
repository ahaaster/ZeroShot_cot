"""
Prepare the dataset for the prompt framework implementation in this file.
Also construct the bespoke signature here that is used in the first phase of zero-shot CoT
"""
import json
import pandas as pd
from pathlib import Path

HERE = Path("dataset/human-level/LargerAnimal")
RAW_SET = HERE / "larger_animal.json"

def prepare_dataset():
    with open(RAW_SET) as file:
        dataset = json.load(file)
    
    dataset = dataset["examples"]
    result = []
    for _, data in dataset.items():
        question = "Which of these animals is the larger one?"
        choices = data["input"].split(", ")
        label = data["output"]

        result.append({
            "label": label,
            "question": question,
            "choices": choices
        })
       
    return result    

if __name__ == "__main__":
    result = prepare_dataset()
    df = pd.DataFrame(result)
    df.to_csv(HERE / "data.csv", index=False)

