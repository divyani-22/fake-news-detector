import joblib
import re

model = joblib.load("model_india.pkl")
vectorizer = joblib.load("vectorizer_india.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

text = "India launches first manned mission to Mars, completes round trip in 8 months"
cleaned = clean_text(text)
vec = vectorizer.transform([cleaned])

prediction = model.predict(vec)[0]
probability = model.predict_proba(vec)[0]

print("Cleaned text:", cleaned)
print("Prediction:", "Real" if prediction == 1 else "Fake")
print("Probability [Fake, Real]:", probability)