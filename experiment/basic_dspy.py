import dspy
import pandas as pd
from pathlib import Path
from dspy import LM, Evaluate, Predict
from dspy.evaluate.metrics import answer_exact_match, answer_passage_match

from .dataset import Dataset
from .utils import save_results, save_score
from .evaluation import Metric, Decoder


def basic_dspy(
    dataset: Dataset,
    *,
    metric_name: str,
    decode: bool = False,
    record_results: bool = False,
    **kwargs,
):
    method = kwargs["method"]

    sig = dataset.create_signature()
    prompter = Predict(signature=sig)

    output_name = "answer"
    # # signature = f"{dataset.get_input_names()} -> {output_name}"  # Maybe this causes errors
    # signature = dspy.Signature(f"{dataset.get_input_names()} -> {output_name}")
    # prompter = Predict(signature=signature)

    decoder = Decoder(dataset.answer_format, kwargs["greedy_first"]) if decode else None
    metric = Metric(metric_name, dataset.label_name, output_name, decoder)

    prompt_that_shit = Evaluate(
        devset=dataset,
        metric=metric,
        num_threads=16,
        display_progress=True,
        display_table=kwargs["display_table"],
        # return_all_scores=True,  # return_outputs geeft al all_scores inbegrepen
        return_outputs=True,
        provide_traceback=True,
        max_errors=10_000,
    )

    avg_score, results = prompt_that_shit(prompter)

    if not record_results:
        # print(results)
        return

    results_df = pd.DataFrame(results)
    results_dir = Path("results") / method / dataset.name

    save_results(results_df, results_dir, dataset.model_name)
    save_score(
        score=avg_score,
        model_name=dataset.model_name,
        method=method,
        metric_name=metric_name,
        dataset_name=dataset.name,
    )
