import dspy
import time
from pathlib import Path

from utils import Num, load_dataset
from secret import secret


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
        
        # Now construct the second signature
        conclusion_sig = dspy.Signature(f"{inpoets}, {self.reason_prefix} -> {outputs}")
        self.predict = dspy.Predict(conclusion_sig)

    def forward(self, **kwargs):
        # First retrieve the reasoning, then add this to the original query for a final answer prompt
        query = self.query_gen(**kwargs)
        kwargs[self.reason_prefix] = query[self.reason_prefix]
        return self.predict(**kwargs)


def main(chosen_model, method, file_path: Path):
    data = load_dataset(file_path=file_path)
    
    dataset = []
    try:
        for label, question, choices in data.values[:4]:
            dataset.append(
                dspy.Example(
                    question=question,
                    answer=label,
                    choices=choices
                ).with_inputs("question", "choices")
            )
        inpoets = "question, choices"
    except: 
        for label, question in data.values:
            dataset.append(
                dspy.Example(
                    question=question,
                    answer=label,
                ).with_inputs("question")
            )
        inpoets = "question"

    lm = dspy.LM(chosen_model, max_tokens=2000, api_key=API_KEY)
    dspy.configure(lm=lm)
    
    if method == "zero-shot":
        prompter = dspy.Predict(f"{inpoets} -> answer")
    elif method == "default":
        prompter = CoT("question, choices -> answer")
    else: 
        prompter = Reasoning(
            inpoets=inpoets, 
            outputs="answer", 
            reasoning_hint="Avadra Kadavra!"
        )

    for query in dataset[2:3]:
        response = prompter(**query.inputs())
        print(response)
        print(query.labels())
        print()



    # x = lm.inspect_history(n=1)
    # print(x)
    
    # for _, row in data.iterrows():
    #     question = row["question"]
    #     choices = row["choices"]
    #     label = row["labels"]

    #     reasoning = first_step(query=question, choices=choices)
    #     print(reasoning)
    #     final_question = f"{question} \n\n{reasoning.output}"
    #     print(final_question)

    #     response = second_step(query=final_question)
    #     print(response.output)
    #     print(f"{label = }")
        
if __name__ == "__main__":
    LLM_MODEL = ["openai/gpt-4", "openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-3.5-turbo"][-1]
    METHODS = ["zero-shot", "default", "eigen"][-1]
    # file_path = Path("dataset/zero-shot_cot/CommonsenseQA/data.csv")
    file_path = Path("dataset/zero-shot_cot/StrategyQA/data.csv")
    
    main(
        chosen_model=LLM_MODEL, 
        method=METHODS, 
        file_path=file_path
    )
