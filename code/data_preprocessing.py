import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from pandas import DataFrame as df
import os

# downloads dataset to local machine
dataset_dir = kagglehub.dataset_download("jp797498e/twitter-entity-sentiment-analysis")

train_path = os.path.join(dataset_dir, "twitter_training.csv")
valid_path = os.path.join(dataset_dir, "twitter_validation.csv")

train_df = pd.read_csv(train_path, encoding="utf-8", engine="python", on_bad_lines="skip")
train_df.columns = ["tweet_id", "entity", "sentiment", "tweet_content"]
valid_df = pd.read_csv(valid_path, encoding="utf-8", engine="python", on_bad_lines="skip")
valid_df.columns = ["tweet_id", "entity", "sentiment", "tweet_content"]

print("Training set preview:\n", train_df.head())
print("\nValidation set preview:\n", valid_df.head())
