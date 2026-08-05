import pandas as pd

from src.configs.salary_predict.salary_experiment_config import (
    get_experiment_config,
)

df = pd.read_parquet(
    "artifacts/salary_dataset_splits/latest/train.parquet"
)

config = get_experiment_config("E1")

print("E1 categorical features:")
print(config.categorical_features)

for col in config.categorical_features:

    print("\n" + "=" * 60)
    print(col)

    print("dtype:", df[col].dtype)
    print("null count:", df[col].isna().sum())

    print("sample:")
    print(df[col].head(10).tolist())