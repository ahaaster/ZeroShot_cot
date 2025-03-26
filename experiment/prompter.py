from dspy import Predict, ChainOfThought, Evaluate
from dspy.evaluate import SemanticF1

from .dataset import Dataset
from .evaluation import Metric


def prompt(
    dataset: Dataset,
    *,
    metric_name: str,
    decode: bool = False,
    record_results: bool = False,
    **kwargs,
):
    dataset = dataset[:20]

    sig = dataset.create_signature()
    prompter = dataset.prompter(signature=sig)
    metric = Metric(metric_name, dataset.label_name)

    prompt_that_shit = Evaluate(
        devset=dataset,
        metric=metric,
        num_threads=16,
        display_progress=True,
        display_table=kwargs["display_table"],
        return_outputs=True,
        provide_traceback=True,
        max_errors=10_000,
    )

    avg_score, results = prompt_that_shit(prompter)

    if not record_results:
        # print(results)
        return

    method = kwargs["method"]

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
