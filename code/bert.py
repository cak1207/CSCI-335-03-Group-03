import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

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

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    small_train_text = train_text[:1000]
    small_y_train = y_train[:1000]

    print("Embedding small training subset for GridSearch...")
    X_small = embedder.encode(small_train_text, batch_size=128, show_progress_bar=True)


    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    param_grid = [
        {
            "clf__solver": ["saga"],
            "clf__penalty": ["l2", "l1"],
            "clf__C": [0.01, 0.1, 1, 10]
        },
        {
            "clf__solver": ["saga"],
            "clf__penalty": ["elasticnet"],
            "clf__l1_ratio": [0.3, 0.5, 0.7],
            "clf__C": [0.01, 0.1, 1]
        },
        {
            "clf__solver": ["liblinear"],
            "clf__penalty": ["l1", "l2"],
            "clf__C": [0.01, 0.1, 1, 10]
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
    print("Best grid score:", grid.best_score_)
    print("Best parameters:", grid.best_params_)


    best_params = grid.best_params_
    clf_params = {k.replace("clf__", ""): v for k, v in best_params.items() if k.startswith("clf__")}

    X_train = embedder.encode(train_text, show_progress_bar=True)
    X_valid = embedder.encode(valid_text, show_progress_bar=True)

    # Build final pipeline + fit on training text
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, **clf_params))
    ])

    print("Retraining final model on full training data with best params...")
    final_pipe.fit(X_train, y_train)

    # evaluate
    preds = final_pipe.predict(X_valid)
    print(classification_report(y_valid, preds, target_names=encoder.classes_))

    # Save final model
    os.makedirs("bert_sklearn_model", exist_ok=True)
    joblib.dump(final_pipe, "bert_sklearn_model/classifier_pipeline.joblib")

    print("Model and artifacts saved to bert_sklearn_model/")


if __name__ == "__main__":
    train_sklearn_bert_model()
