from data_preprocessing import load_and_clean_data, encode_labels, vectorize_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np
import pickle

df = load_and_clean_data()

y, encoder = encode_labels(df)

x = df['tweet_content']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


x_train_tfidf, x_test_tfidf, tfidf = vectorize_text(x_train, x_test)

svc = SVC(max_iter=100000)


np.random.seed(42)

range = np.random.choice(x_train_tfidf.shape[0], size=15000, replace=False)
x_train_small = x_train_tfidf[range]
y_train_small = y_train[range]

param_grid = [
    {
        "kernel": ["linear"],
        "C": [ 0.1, 1, 3, 5, 10],
        "class_weight": ["balanced"]
    },
    {
        "kernel": ["rbf"],
        "C": [ 0.1, 1, 3, 5, 10],
        "gamma": ["scale", "auto", 0.01, 0.1, 1],
        "class_weight": ["balanced"]
    },
    {
        "kernel": ["poly"],
        "C": [ 0.1, 1, 3, 5, 10],
        "degree": [2, 3],
        "gamma": ["scale", "auto", 0.01, 0.1, 1],
        "class_weight": ["balanced"]
    },
    {
        "kernel": ["sigmoid"],
        "C": [ 0.1, 1, 3, 5, 10],
        "gamma": ["scale", "auto", 0.01, 0.1, 1],
        "class_weight": ["balanced"]
    }
]

grid = GridSearchCV(
    estimator=svc,
    param_grid=param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=2
)


#grid.fit(x_train_tfidf, y_train)
grid.fit(x_train_small, y_train_small)

y_pred = grid.predict(x_test_tfidf)

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("\nBest Parameters:")
print(grid.best_params_)

print("\nBest CV Score:", grid.best_score_)

weighted_f1 = f1_score(y_test, y_pred, average='weighted')
print("Best Test Weighted F1:", weighted_f1)

print("Best Test Accuracy:", accuracy_score(y_test, y_pred))

print("\nPrecision (weighted):", precision)

print("Recall (weighted):", recall)



print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

best_svc = grid.best_estimator_

with open("best_svc_model.pkl", "wb") as f:
    pickle.dump(best_svc, f)