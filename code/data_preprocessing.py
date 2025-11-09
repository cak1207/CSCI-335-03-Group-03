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

print("Before cleanup:")
print()

print(train_df.info())


#some data has missing values for content so I have to drop them
train_df = train_df.dropna(subset=['tweet_content'])
print()

print("After cleanup:")

print(train_df.info())

print(train_df['sentiment'].value_counts())

encoder = LabelEncoder()

label_encoded = encoder.fit_transform(train_df['sentiment'])
print()

print("Label encoding :")
print(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

print("some  encoded labels:", label_encoded[:50])

x = train_df['tweet_content']
y = label_encoded 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=10000, stop_words='english')

x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)

svm_model = LinearSVC(class_weight='balanced', random_state=42)

# Train the model
svm_model.fit(x_train_tfidf, y_train)

y_prediction = svm_model.predict(x_test_tfidf)

print()
print("Report:")
print( classification_report(y_test, y_prediction, target_names=encoder.classes_))
