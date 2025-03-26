import pandas as pd
from pathlib import Path


def get_saved_data(dir_path: Path, model_name: str) -> pd.DataFrame:
    """Function to obtain data specified by a directory and model name.
    If no such saved results exist, create the directory and return an empty DataFrame
    """
    model_name = convert_model_filename(model_name)
    save_paths: list = fetch_datasets(dir_path, model_name)

    if not save_paths:
        dir_path.mkdir(parents=True, exist_ok=True)
        return pd.DataFrame()

    return pd.read_csv(save_paths[0])


def save_results(df: pd.DataFrame, dir_path: Path, model_name: str) -> None:
    model_name = convert_model_filename(model_name)
    save_path = Path(dir_path) / model_name
    save_path = save_path.with_suffix(".csv")
    df.to_csv(save_path, index=False)


def save_score(
    score: float, model_name: str, method: str, metric_name: str, dataset_name: str
) -> None:
    file_path = Path("results/scores.csv")
    df = pd.read_csv(file_path, index_col=[0, 1], header=[0, 1])

    df.loc[(dataset_name, model_name), (method, metric_name)] = score
    df.to_csv(file_path)


def fetch_datasets(dir_path: Path, file_name: str = "*") -> list[Path]:
    return sorted(dir_path.glob(f"**/{file_name}.csv"))


def get_dir_name(file_path: Path) -> str:
    return file_path.parent.stem


def convert_model_filename(model_name: str) -> str:
    return model_name.replace(".", "_").replace(":", "=")


def revert_model_filename(file_name: Path) -> str:
    file_name = file_name.stem
    return file_name.replace("_", ".").replace("=", ":")
