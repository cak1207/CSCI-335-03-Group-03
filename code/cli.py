import sys
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import bert
import svm

def main():
    print("Sentiment Analysis of Twitter Data:")
    print("Usage: python code/cli.py model_type train/test path_1 path_2")

    if len(sys.argv) != 5:
        print("Error: Command needs 5 arguments")
        return

    model_type = sys.argv[1]
    if model_type != "svm" and model_type != "bert":
        print("Error: argument 1, '" + model_type + "' must be either 'svm' or 'bert'")
        return
    action_type = sys.argv[2]   # train or test
    if action_type != "train" and action_type != "test":
        print("Error: argument 2, '" + action_type + "' must be either 'train' or 'test'")
        return
    path_1 = sys.argv[3]
    path_2 = sys.argv[4]


    if action_type == "train":
        if model_type == "svm":
            svm.train_sklearn_svn_model(path_1, path_2)
        if model_type == "bert":
            bert.train_sklearn_bert_model()


    if action_type == "test":
        output_path = os.path.join("data", path_2) 

        df = pd.read_csv(path_1, encoding="utf-8", engine="python", on_bad_lines="skip")
        df.columns = ["tweet_id", "entity", "sentiment", "tweet_content"]

        if model_type == "svm":
            model_path = "data/best_svm_pipeline.pkl"
        elif model_type == "bert":
            model_path = "data/best_bert_pipeline.pkl"
        else:
            print("Error: model_type must be 'svm' or 'bert'")
            return

        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)

        model = pipeline.get("model")
        label_encoder = pipeline.get("label_encoder")

        if model_type == "svm":
            tfidf = pipeline.get("tfidf")
            X = tfidf.transform(df["tweet_content"])
            preds = model.predict(X)


        if model_type == "bert":
            bert.test_bert_model()

        df["predicted"] = label_encoder.inverse_transform(preds)

        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Predictions saved to {output_path}")

        y_true = label_encoder.transform(df["sentiment"])
        y_pred = preds

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        prec = precision_score(y_true, y_pred, average="weighted")
        rec = recall_score(y_true, y_pred, average="weighted")
        print(f"Accuracy: {acc}")
        print(f"Weighted F1: {f1}")
        print(f"Weighted Precision: {prec}")
        print(f"Weighted Recall: {rec:}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))


if __name__ == "__main__":
    main()
