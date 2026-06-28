import pandas as pd
import re
from my_examples import my_data

# Load IFND dataset
data = pd.read_csv("IFND.csv", encoding="latin1")

# Clean function (same as before)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["clean_text"] = data["Statement"].apply(clean_text)
data["label"] = data["Label"].apply(lambda x: 1 if str(x).strip().upper() == "TRUE" else 0)

# Turn your custom examples into the same format
my_df = pd.DataFrame(my_data)
my_df["clean_text"] = my_df["text"].apply(clean_text)
my_df["label"] = my_df["label"].apply(lambda x: 1 if x.strip().lower() == "real" else 0)

print("IFND rows:", len(data))
print("Your custom rows:", len(my_df))

# Combine both datasets
combined = pd.concat([data[["clean_text", "label"]], my_df[["clean_text", "label"]]], ignore_index=True)

print("Combined total rows:", len(combined))
print("\nLabel balance after combining:")
print(combined["label"].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

X_train, X_test, y_train, y_test = train_test_split(
    combined["clean_text"], combined["label"], test_size=0.2, random_state=42, stratify=combined["label"]
)

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)
print("\nAccuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, target_names=["Fake", "Real"]))

joblib.dump(model, "model_india.pkl")
joblib.dump(vectorizer, "vectorizer_india.pkl")
print("\nCombined model saved (overwriting model_india.pkl)!")