import dspy
import time
import pandas as pd
from pathlib import Path
from dspy.evaluate import SemanticF1

from utils import Num, load_dataset
from secret import secret


LLM_MODEL = ["openai/gpt-3.5-turbo", "openai/gpt-4", "openai/gpt-4o-mini", "openai/gpt-4o"]
METHODS = ["zero-shot", "cot_default", "cot_imitation", "zero-shot-cot"]
SCORING_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]    
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
        
        # answer_field = dspy.OutputField(prefix="Therefore the answer is", desc="The answer is a number")
        conclusion_sig = dspy.Signature(f"{inpoets}, {self.reason_prefix} -> {outputs}")
        self.predict = dspy.Predict(conclusion_sig)

    def forward(self, **kwargs):
        # First retrieve the reasoning, then add this to the original query for a final answer prompt
        query = self.query_gen(**kwargs)
        kwargs[self.reason_prefix] = query[self.reason_prefix]
        return self.predict(**kwargs)   

def get_prompter(method_name, inpoets, outputs):
    method_dict = {
        METHODS[0]: dspy.Predict(f"{inpoets} -> {outputs}"),
        METHODS[1]: dspy.ChainOfThought(f"{inpoets} -> {outputs}"),
        METHODS[2]: CoT(f"{inpoets} -> {outputs}"),
        METHODS[-1]: Reasoning(
            inpoets=inpoets, 
            outputs=outputs, 
            # reasoning_hint="Avada Kadavra!"
        )
    }
    return method.get(method_name, None)


def main(chosen_model, method, file_path: Path, track_scores: bool = False):
    lm = dspy.LM(chosen_model, max_tokens=2000, api_key=API_KEY)
    dspy.configure(lm=lm)
    
    dataset_name = file_path.parent.stem
    data = load_dataset(file_path=file_path)
    
    dataset = []
    try:  # Hacky try-except block to create dspy.Example dataset depending on problem set, to be remade later
        for label, question, choices in data.values:
            dataset.append(
                dspy.Example(
                    response=label,
                    question=question,
                    choices=choices
                ).with_inputs("question", "choices")
            )
        inpoets = "question, choices"
        outputs = "response"
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

    prompter = get_prompter(method_name=method)
    if prompter is None:  # sanity check
        return

    
    for threshold in SCORING_THRESHOLDS:
        def semantic_scoring(example, prediction):
            score = SemanticF1(decompositional=True)(example, prediction)
            return 1 if score > threshold else 0

        # metric = SemanticF1(decompositional=True)
        metric = semantic_scoring
        
        evaluate = dspy.Evaluate(
            devset=dataset,
            metric=metric,
            num_threads=16,  # Adjust to specs of local CPU available
            display_progress=True,
            # display_table=4,
            provide_traceback=True
        )

        score = evaluate(prompter)
        if track_scores:
            update_results(score, chosen_model, dataset_name, threshold, method)
        else:
            print(f"{chosen_model} had an accuracy of {score} % on the {dataset_name} dataset")


def update_results(score, chosen_model, dataset_name, threshold_val, method):
    file_path = Path(f"experiment/{method}.csv")
    df = pd.read_csv(file_path, index_col=[0, 1])
    
    df.loc[(dataset_name, chosen_model), f"{threshold_val}"] = score
    df.to_csv(file_path)
    
# HELPER FUNCTION CONCERNING RECORDING SCORES
def create_results_file(method_name: str = None):
    """This function should only be run once if the csv with recorded results doesn't exist"""
    datasets = Path("dataset/zero-shot_cot").glob("**/*.csv")
    dataset_names = [x.parent.stem for x in sorted(datasets)]
    
    iterables = [dataset_names, LLM_MODEL]
    index = pd.MultiIndex.from_product(iterables, names=["dataset", "model"])
    cols = [str(x) for x in SCORING_THRESHOLDS]
    df = pd.DataFrame(0, index=index, columns=cols)
    
    file_path = Path(f"experiment/{method_name}.csv") if method_name else RESULTS_PATH
    df.to_csv(file_path)


if __name__ == "__main__":
    method = METHODS[0] 
    track_scores = True

    prepped_datasets = Path("dataset/zero-shot_cot").glob("**/data.csv")
    prepped_datasets: list[Path] = sorted(prepped_datasets)
    # file_path: Path = prepped_datasets[0]

    # create_results_file(method_name=method)
    for file_path in prepped_datasets[:]:
        main(LLM_MODEL[3], method, file_path, track_scores)

# Copy pasted constants for easy visual access main() arguments indexing
#   LLM_MODEL = ["openai/gpt-3.5-turbo", "openai/gpt-4", "openai/gpt-4o-mini", "openai/gpt-4o"]
#   METHODS = ["zero-shot", "cot_default", "cot_imitation", "zero-shot-cot"]
#   SCORING_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]    
