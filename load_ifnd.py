import pandas as pd

data = pd.read_csv("IFND.csv", encoding="latin1")

print("Shape:", data.shape)
print("\nColumns:", data.columns.tolist())
print("\nFirst 5 rows:")
print(data.head())

print("\nLabel value counts:")
print(data[data.columns[-1]].value_counts())

for col in data.columns:
    print(col)

    import re

# Clean the Statement text (same approach as before)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["clean_statement"] = data["Statement"].apply(clean_text)

# Convert labels: TRUE -> 1, Fake -> 0
data["label"] = data["Label"].apply(lambda x: 1 if x.strip().upper() == "TRUE" else 0)

print("\nLabel counts after conversion:")
print(data["label"].value_counts())

print("\nSample cleaned statement:")
print(data["clean_statement"].iloc[0])

from sklearn.model_selection import train_test_split

X = data["clean_statement"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print("\nTraining label balance:")
print(y_train.value_counts())

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\nTraining matrix shape:", X_train_vec.shape)
print("Testing matrix shape:", X_test_vec.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)

print("\nModel accuracy:", accuracy)
print("\nDetailed report:")
print(classification_report(y_test, predictions, target_names=["Fake", "Real"]))


import joblib

joblib.dump(model, "model_india.pkl")
joblib.dump(vectorizer, "vectorizer_india.pkl")

print("\nIndia model and vectorizer saved!")

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")