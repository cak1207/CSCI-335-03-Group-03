import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from pandas import DataFrame as df
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report



import kagglehub
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


def load_and_clean_data():
    # downloads dataset to local machine
    dataset_dir = kagglehub.dataset_download("jp797498e/twitter-entity-sentiment-analysis")

    train_path = os.path.join(dataset_dir, "twitter_training.csv")
    valid_path = os.path.join(dataset_dir, "twitter_validation.csv")

    train_df = pd.read_csv(train_path, encoding="utf-8", engine="python", on_bad_lines="skip")
    train_df.columns = ["tweet_id", "entity", "sentiment", "tweet_content"]

    # Drop missing text entries
    train_df = train_df.dropna(subset=['tweet_content'])

    return train_df


def encode_labels(df):
    encoder = LabelEncoder()
    labels = encoder.fit_transform(df['sentiment'])
    return labels, encoder


def vectorize_text(train_text, test_text=None):
    tfidf = TfidfVectorizer(max_features=10000, stop_words="english")

    train_vectors = tfidf.fit_transform(train_text)
    test_vectors = tfidf.transform(test_text) if test_text is not None else None

    return train_vectors, test_vectors, tfidf

if __name__ == "__main__":
    print()