from data_preprocessing import load_and_clean_data, encode_labels, vectorize_text
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,accuracy_score


df = load_and_clean_data()

y, encoder = encode_labels(df)

x = df['tweet_content']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


x_train_tfidf, x_test_tfidf, tfidf = vectorize_text(x_train, x_test)


svm_model = LinearSVC(class_weight="balanced", random_state=42)
svm_model.fit(x_train_tfidf, y_train)

# 6. Predict
y_pred = svm_model.predict(x_test_tfidf)

results = []

C_values = [0.5, 1.0, 1.5, 2.0, 2.5,
            3.0, 3.5, 4.0, 4.5, 5.0,
            5.5, 6.0, 6.5, 7.0, 7.5,
            8.0, 8.5, 9.0, 9.5, 10.0]

#we are testing C values to check which ones fit the best
for c in C_values:
    print("\nTesting C =", c)
    svm_model = LinearSVC(C=c, class_weight="balanced", random_state=42)
    svm_model.fit(x_train_tfidf, y_train)
    y_pred = svm_model.predict(x_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    results.append((c, acc))
    print("Accuracy:", acc)


sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

for C, score in sorted_results[:5]:
    print(f"C={C:.1f}, Score={score}")

    #print("\nClassification Report:")
    #print(classification_report(y_test, y_pred, target_names=encoder.classes_))