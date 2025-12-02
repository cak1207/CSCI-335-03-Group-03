import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.model_selection import train_test_split
from data_preprocessing import load_and_clean_data, encode_labels


def train_sklearn_bert_model():
    df = load_and_clean_data()

    # encode df
    labels, encoder = encode_labels(df)
    df["label"] = labels

    # splits
    train_text, valid_text, y_train, y_valid = train_test_split(
        df["tweet_content"].tolist(),
        df["label"].tolist(),
        test_size=0.1,
        random_state=42
    )

    model = SentenceTransformer("all-mpnet-base-v2")

    print("Embedding training data...")
    X_train = model.encode(train_text, show_progress_bar=True)

    print("Embedding validation data...")
    X_valid = model.encode(valid_text, show_progress_bar=True)

    # Train classifier
    # needs grid search for optimal hyperparameters
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="saga", C=1.0)
    clf.fit(X_train, y_train)

    # evaluate
    preds = clf.predict(X_valid)
    print(classification_report(y_valid, preds, target_names=encoder.classes_))

    # Save model
    os.makedirs("bert_sklearn_model", exist_ok=True)
    joblib.dump(clf, "bert_sklearn_model/classifier.joblib")
    model.save("bert_sklearn_model/sentence_transformer")
    joblib.dump(encoder, "bert_sklearn_model/label_encoder.joblib")

    print("Model saved to bert_sklearn_model/")


if __name__ == "__main__":
    train_sklearn_bert_model()
