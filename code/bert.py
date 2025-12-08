import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from data_preprocessing import load_and_clean_data, encode_labels
import pickle

def train_sklearn_bert_model(train_path=None, valid_path=None):
    train_df, valid_df = load_and_clean_data(train_path, valid_path)

    train_labels, encoder = encode_labels(train_df)
    train_df["label"] = train_labels
    valid_df["label"] = encoder.transform(valid_df["sentiment"])

    train_text = train_df["tweet_content"].tolist()
    y_train = train_df["label"].tolist()

    valid_text = valid_df["tweet_content"].tolist()
    y_valid = valid_df["label"].tolist()

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
    best_model = grid.best_estimator_

    X_valid = embedder.encode(valid_text, show_progress_bar=True)
    y_pred = best_model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    weighted_f1 = f1_score(y_valid, y_pred, average='weighted')
    weighted_precision = precision_score(y_valid, y_pred, average='weighted')
    weighted_recall = recall_score(y_valid, y_pred, average='weighted')

    print("\nBest Parameters:")
    print(grid.best_params_)

    print("\nBest CV Score:", grid.best_score_)
    print("Best Test Weighted F1:", weighted_f1)
    print("Best Test Accuracy:", accuracy)
    print("Weighted Precision:", weighted_precision)
    print("Weighted Recall:", weighted_recall)
    

    # save model
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "data")
    DATA_DIR = os.path.abspath(DATA_DIR)

    model_path = os.path.join(DATA_DIR, "best_bert_pipeline.pkl")

    # needs to pickle pipeline w/ encoder and model
    pipeline = {
        "model": best_model,
        "label_encoder": encoder,
        "embedder": "all-MiniLM-L12-v2"
    }
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print("\nSaved pipeline to:", model_path)



    






if __name__ == "__main__":
    train_sklearn_bert_model()
