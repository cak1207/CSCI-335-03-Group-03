import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from data_preprocessing import load_and_clean_data, encode_labels
import pickle

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

    # "bert-base-uncased"
    embedder = SentenceTransformer("all-MiniLM-L12-v2")
    np.random.seed(42)

    indices = np.random.choice(len(train_text), size=15000, replace=False)
    small_train_text = [train_text[i] for i in indices]
    small_y_train = [y_train[i] for i in indices]
    X_small = embedder.encode(small_train_text, show_progress_bar=True)


    print("Embedding small training subset for GridSearch...")

    pipe = Pipeline([
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=42))
    ])

    param_grid = [
        {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 20, 40],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 2],
            "clf__max_features": ["sqrt", "log2"],
        }
    ]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # grid
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        refit=True
    )

    grid.fit(X_small, small_y_train)
    print("Best parameters:", grid.best_params_)
    print("Best grid score:", grid.best_score_)

    best_model = grid.best_estimator_

    # save model
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "data")
    DATA_DIR = os.path.abspath(DATA_DIR)

    model_path = os.path.join(DATA_DIR, "best_bert_model.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print("Model saved")

    X_valid = embedder.encode(valid_text, show_progress_bar=True)
    y_pred = best_model.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_pred)
    weighted_f1 = f1_score(y_valid, y_pred, average='weighted')
    weighted_precision = precision_score(y_valid, y_pred, average='weighted')
    weighted_recall = recall_score(y_valid, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_valid, y_pred))


    best_params = grid.best_params_
    clf_params = {k.replace("clf__", ""): v for k, v in best_params.items() if k.startswith("clf__")}

    X_train = embedder.encode(train_text, show_progress_bar=True)



    # Build final pipeline + fit on training text
    final_pipe = Pipeline([
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=42, **clf_params))
    ])

    print("Retraining final model on full training data with best params...")
    final_pipe.fit(X_train, y_train)

    y_pred = final_pipe.predict(X_valid)

    print("after final pipeline: \n")
    accuracy = accuracy_score(y_valid, y_pred)
    weighted_f1 = f1_score(y_valid, y_pred, average='weighted')
    weighted_precision = precision_score(y_valid, y_pred, average='weighted')
    weighted_recall = recall_score(y_valid, y_pred, average='weighted')

    print("Best parameters:", clf_params)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_valid, y_pred, target_names=encoder.classes_))


if __name__ == "__main__":
    train_sklearn_bert_model()
