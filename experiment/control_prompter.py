import pandas as pd
from tqdm import tqdm
from dspy import Example, LM
from pathlib import Path

from .dataset import Dataset
from .utils import convert_model_filename, fetch_datasets


def create_prompt(data: Example, input_keys: list[str], join_string: str = "\n") -> str:
    prompt = [data[key] for key in input_keys]
    return join_string.join(prompt)


def prompt_control(lm: LM, dataset: Dataset, model_name: str, record_results: bool):
    # Load previous dataset paths
    results_dir = Path("results/control") / dataset.name
    model_name = convert_model_filename(model_name)
    results_path = fetch_datasets(results_dir, model_name)

    # Check if the previous dataset is empty or not
    if results_path:
        df_results = pd.read_csv(results_path[0])
        prompts_recorded = len(df_results)
    else:
        results_dir.mkdir(parents=True, exist_ok=True)
        prompts_recorded = 0
        df_results = pd.DataFrame()

    # Determine portion of dataset to be prompted
    n_unprompted: int = len(dataset) - prompts_recorded
    if n_unprompted == 0:
        return

    # Create batches of to be prompted queries
    unrecorded: Dataset = dataset[-n_unprompted:]
    batch_n = 25
    batches: list[list[example]] = [
        unrecorded[i : i + batch_n] for i in range(0, len(unrecorded), batch_n)
    ]

    # Function part that actually prompts
    for data_batch in tqdm(batches):
        responses = []
        for example in tqdm(data_batch):
            prompt: str = create_prompt(example, dataset.input_names)

            response: list[str] = lm(messages=[{"role": "user", "content": prompt}])
            responses.append(
                {
                    "prompt": prompt,
                    "response": response[0],
                    "ground_truth": example[dataset.label_name],
                }
            )

        batch_df = pd.DataFrame(responses)
        df_results = pd.concat([df_results, batch_df])

        if not record_results:
            print(df_results.tail(3))
            continue

        df_results.to_csv(f"{results_dir}/{model_name}.csv", index=False)
