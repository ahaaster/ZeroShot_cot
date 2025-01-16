import dspy
import time
import pandas as pd
from pathlib import Path
from dspy.evaluate import SemanticF1

from utils import Num, load_dataset
from secret import secret


LLM_MODEL = ["openai/gpt-3.5-turbo", "openai/gpt-4", "openai/gpt-4o-mini", "openai/gpt-4o"]
METHODS = ["zero-shot", "default", "eigen"]
SCORING_THRESHOLDS = [0.6, 0.7, 0.8]    
RESULTS_PATH = Path("experiment/results.csv")

API_KEY = secret if secret else ""


class CoT(dspy.Module):
    """This is how DSPy's ChainOfThought module works"""
    def __init__(self, signature, rationale_type=None, **config):
        super().__init__()
        signature = dspy.signatures.signature.ensure_signature(signature)
        
        prefix = "Reasoning: Let's think step by step in order to"
        desc = "${reasoning}"
        rationale_type = rationale_type or dspy.OutputField(prefix=prefix, desc=desc)
        extended_signature = signature.prepend("reasoning", rationale_type, type_=str)

        self.predict = dspy.Predict(extended_signature, **config)
       
    def forward(self, **kwargs):
        return self.predict(**kwargs)

class Reasoning(dspy.Module):
    # Supply a decomposed signature in the form of '*inpoets -> *outputs'
    # First retrieve a rationale to create a final query, 
    #   then use this rationale with the original inpoet to generate the answer in a second prompt
    def __init__(self, inpoets: str, outputs: str, reasoning_hint: str = "Let's think step by step"):
        self.reason_prefix = "reasoning" 

        reasoning_field = dspy.OutputField(prefix=f"{self.reason_prefix}: {reasoning_hint}")
        reasoning_sig = dspy.Signature(f"{inpoets} ->").prepend(self.reason_prefix, reasoning_field, type_=str)
        self.query_gen = dspy.Predict(reasoning_sig)
        
        conclusion_sig = dspy.Signature(f"{inpoets}, {self.reason_prefix} -> {outputs}")
        self.predict = dspy.Predict(conclusion_sig)

    def forward(self, **kwargs):
        # First retrieve the reasoning, then add this to the original query for a final answer prompt
        query = self.query_gen(**kwargs)
        kwargs[self.reason_prefix] = query[self.reason_prefix]
        return self.predict(**kwargs)   

def main(chosen_model, method, file_path: Path, threshold):
    dataset_name = file_path.parent.stem
    data = load_dataset(file_path=file_path)
    
    dataset = []
    try:
        for label, question, choices in data.values[:4]:
            dataset.append(
                dspy.Example(
                    answer=label,
                    question=question,
                    choices=choices
                ).with_inputs("question", "choices")
            )
        inpoets = "question, choices"
        outputs = "answer"
    except: 
        for label, question in data.values:
            dataset.append(
                dspy.Example(
                    response=label,
                    question=question,
                ).with_inputs("question")
            )
        inpoets = "question"
        outputs = "response"

    lm = dspy.LM(chosen_model, max_tokens=2000, api_key=API_KEY)
    dspy.configure(lm=lm)
    
    if method == "zero-shot":
        prompter = dspy.Predict(f"{inpoets} -> {outputs}")
    elif method == "default":
        prompter = CoT(f"{inpoets} -> {outputs}")
    else: 
        prompter = Reasoning(
            inpoets=inpoets, 
            outputs=outputs, 
            # reasoning_hint="Avadra Kadavra!"
        )

    def semantic_scoring(example, prediction):
        score = SemanticF1(decompositional=True)(example, prediction)
        return 1 if score > threshold else 0

    # metric = SemanticF1(decompositional=True)
    metric = semantic_scoring
    
    evaluate = dspy.Evaluate(
        devset=dataset[:100],
        metric=metric,
        num_threads=8,
        display_progress=True,
        # display_table=4,
        provide_traceback=True
    )

    score = evaluate(prompter)
    # print(f"Our estimated accuracy is: {score} %")
    update_results(score, chosen_model, dataset_name, threshold)


def update_results(score, chosen_model, dataset_name, threshold_val):
    df = pd.read_csv(RESULTS_PATH, index_col=[0, 1])
    df.loc[(dataset_name, chosen_model), f"{threshold_val}"] = score
    df.to_csv(RESULTS_PATH)
    

def create_results_file():
    paths = Path("dataset/zero-shot_cot").glob("**/*.csv")
    dataset_names = [x.parent.stem for x in sorted(paths)]
    
    iterables = [dataset_names, LLM_MODEL]
    index = pd.MultiIndex.from_product(iterables, names=["dataset", "model"])
    cols = [f">{x}" for x in SCORING_THRESHOLDS]
    df = pd.DataFrame(0, index=index, columns=SCORING_THRESHOLDS)
    
    df.to_csv(RESULTS_PATH)


if __name__ == "__main__":
    # create_results_file()
    # file_path = Path("dataset/zero-shot_cot/CommonsenseQA/data.csv")
    # file_path = Path("dataset/zero-shot_cot/StrategyQA/data.csv")
    
    prepped_datasets = Path("dataset/zero-shot_cot").glob("**/data.csv")
    prepped_datasets = sorted(prepped_datasets)
    file_path = prepped_datasets[4]
    main(chosen_model=LLM_MODEL[0], method=METHODS[-1], file_path=file_path, threshold=SCORING_THRESHOLDS[0])
    # print(update_results(0,0,0,0))
