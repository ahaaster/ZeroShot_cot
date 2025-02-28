import dspy
import pandas as pd
from pathlib import Path
from dspy import LM, Evaluate, Predict
from dspy.evaluate.metrics import answer_exact_match, answer_passage_match

from .dataset import Dataset
from .utils import save_results
from .evaluation import Metric, Decoder, exact_match, exact_match_lower


def basic_dspy(
    dataset: Dataset, model_name: str, *, record_results: bool = False, **kwargs
):
    # Check if results have been recorded already
    results_dir = Path("results/basic") / dataset.name
    if results_dir.exists():
        return

    output_name = "answer"
    signature = f"{dataset.get_input_names()} -> {output_name}"
    prompter = Predict(signature=signature)

    decoder = Decoder(answer_format="mc", greedy_first=False)
    metric = Metric(exact_match, dataset.label_name, output_name, decoder)

    prompt_that_shit = Evaluate(
        devset=dataset[:10],
        metric=metric,
        num_threads=8,
        display_progress=True,
        display_table=True,
        return_all_scores=True,
        return_outputs=True,
        provide_traceback=True,
    )

    avg_score, results, scores = prompt_that_shit(prompter)

    if not record_results:
        return

    results_df = pd.DataFrame(results)
    save_results(results_df, results_dir, model_name)
