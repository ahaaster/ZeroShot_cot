from pathlib import Path


def fetch_datasets(dir_path: Path, file_name: str = "*") -> list[Path]:
    return sorted(dir_path.glob(f"**/{file_name}.csv"))


def get_dir_name(file_path: Path) -> str:
    return file_path.parent.stem


def convert_model_filename(model_name: str) -> str:
    return model_name.replace(".", "_").replace(":", "=")


def revert_model_filename(file_name: Path) -> str:
    file_name = file_name.stem
    return file_name.replace("_", ".").replace("=", ":")
