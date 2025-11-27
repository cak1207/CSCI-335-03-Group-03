from data_preprocessing import load_and_clean_data, encode_labels, vectorize_text
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,accuracy_score,f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

df = load_and_clean_data()

y, encoder = encode_labels(df)

x = df['tweet_content']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


x_train_tfidf, x_test_tfidf, tfidf = vectorize_text(x_train, x_test)


svc = LinearSVC(max_iter=100000)

param_grid = [
    {
        "loss": ["hinge"],
        "dual": [True],
        "C": [ 0.1, 1, 3, 5, 10],
        "class_weight": ["balanced"]
    },
    {
        "loss": ["squared_hinge"],
        "dual": [True, False],
        "C": [ 0.1, 1, 3, 5, 10],
        "class_weight": ["balanced"]
    }
]


# Grid Search
grid = GridSearchCV(
    estimator= svc,
    param_grid=param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=2
)


grid.fit(x_train_tfidf, y_train)

# Best model predictions
y_pred = grid.predict(x_test_tfidf)

print("\nBest Parameters:")
print(grid.best_params_)

print("\nBest CV Score:" + str(grid.best_score_))
print("\nBest Test Accuracy:" + str(accuracy_score(y_test, y_pred)))

f1_score = f1_score(y_test, y_pred, average='weighted')

print("Best Test Weighted F1:" + str(f1_score))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

