import pandas as pd

# Load both files
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Check what's inside
print("Fake news shape:", fake.shape)
print("Real news shape:", real.shape)

print("\nFake news columns:", fake.columns.tolist())
print("\nFirst fake headline:", fake.iloc[0]["title"])
print("First real headline:", real.iloc[0]["title"])
# Add labels: 0 = fake, 1 = real
fake["label"] = 0
real["label"] = 1

# Combine into one dataset
data = pd.concat([fake, real], ignore_index=True)

# Shuffle the rows so fake/real aren't in separate blocks
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nCombined dataset shape:", data.shape)
print(data["label"].value_counts())
print(data.head())

import re

# Combine title + text for a stronger signal
data["content"] = data["title"] + " " + data["text"]

# Basic cleaning function
def clean_text(text):
    text = text.lower()                          # lowercase everything
    text = re.sub(r"[^a-z\s]", "", text)          # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()      # remove extra spaces
    return text

data["content"] = data["content"].apply(clean_text)

print("\nSample cleaned content:")
print(data["content"].iloc[0][:200])  # show first 200 characters

from sklearn.model_selection import train_test_split

X = data["content"]      # the text (input)
y = data["label"]        # 0 = fake, 1 = real (what we want to predict)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\nTraining matrix shape:", X_train_vec.shape)
print("Testing matrix shape:", X_test_vec.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)

print("\nModel accuracy:", accuracy)

import joblib

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel and vectorizer saved!")