import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from pandas import DataFrame as df


train_df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "jp797498e/twitter-entity-sentiment-analysis",
  "twitter_training.csv",
  pandas_kwargs={"encoding": "ISO-8859-1"}
)

valid_df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "jp797498e/twitter-entity-sentiment-analysis",
  "twitter_validation.csv",
  pandas_kwargs={"encoding": "ISO-8859-1"}
)

print("First 5 training records:\n", train_df.head())
print("\nFirst 5 validation records:\n", valid_df.head())
