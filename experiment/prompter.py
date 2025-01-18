from dspy import Module, Predict, Signature
from dspy import OutputField, InputField

from dspy.signatures.signature import ensure_signature


class Reasoning(Module):
    # Supply a decomposed signature in the form of '*inpoets -> *outputs'
    # First retrieve a rationale to create a final query, 
    #   then use this rationale with the original inpoet to generate the answer in a second prompt
    def __init__(self, inpoets: str, outputs: str, reasoning_hint: str = "Let's think step by step"):
        self.reason_prefix = "reasoning" 

        reasoning_field = OutputField(prefix=f"{self.reason_prefix}: {reasoning_hint}")
        reasoning_sig = Signature(f"{inpoets} ->").prepend(self.reason_prefix, reasoning_field, type_=str)
        self.query_gen = Predict(reasoning_sig)
        
        # answer_field = OutputField(prefix="Therefore the answer is", desc="The answer is a number")
        conclusion_sig = Signature(f"{inpoets}, {self.reason_prefix} -> {outputs}")
        self.predict = Predict(conclusion_sig)

    def forward(self, **kwargs):
        # First retrieve the reasoning, then add this to the original query for a final answer prompt
        query = self.query_gen(**kwargs)
        kwargs[self.reason_prefix] = query[self.reason_prefix]
        return self.predict(**kwargs)   


"""This is DSPy's latest implementation of ChainOfThought module"""
class CoT(Module):
    def __init__(self, signature, rationale_type=None, **config):
        super().__init__()
        signature = ensure_signature(signature)
        
        prefix = "Reasoning: Let's think step by step in order to"
        desc = "${reasoning}"
        rationale_type = rationale_type or OutputField(prefix=prefix, desc=desc)
        extended_signature = signature.prepend("reasoning", rationale_type, type_=str)

        self.predict = Predict(extended_signature, **config)
       
    def forward(self, **kwargs):
        return self.predict(**kwargs)
