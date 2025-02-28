import pandas as pd
from pathlib import Path


def get_saved_data(dir_path: Path, model_name: str) -> pd.DataFrame:
    """Function to obtain data specified by a directory and model name.
    If no such saved results exist, create the directory and return an empty DataFrame
    """
    model_name = convert_model_filename(model_name)
    save_paths: list = fetch_datasets(dir_path, model_name)

    if save_paths:
        return pd.read_csv(save_paths[0])
    else:
        dir_path.mkdir(parents=True, exist_ok=True)
        return pd.DataFrame()


def save_results(df: pd.DataFrame, dir_path: Path, model_name: str) -> None:
    model_name = convert_model_filename(model_name)
    save_path = Path(dir_path) / model_name
    save_path = save_path.with_suffix(".csv")
    df.to_csv(save_path, index=False)


def fetch_datasets(dir_path: Path, file_name: str = "*") -> list[Path]:
    return sorted(dir_path.glob(f"**/{file_name}.csv"))


def get_dir_name(file_path: Path) -> str:
    return file_path.parent.stem


def convert_model_filename(model_name: str) -> str:
    return model_name.replace(".", "_").replace(":", "=")


def revert_model_filename(file_name: Path) -> str:
    file_name = file_name.stem
    return file_name.replace("_", ".").replace("=", ":")
