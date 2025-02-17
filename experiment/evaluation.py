import pandas as pd
from pathlib import Path

from .dataset import Dataset
from .utils import fetch_datasets

path_dataset = Path("dataset/cot/MultiArith")
label_path = fetch_datasets(path_dataset, file_name="data")

df = pd.read_csv(label_path)
df_label = df.loc[:, "label"]
