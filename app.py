from flask import Flask, request, jsonify, render_template
import joblib
import re

app = Flask(__name__)

# Load the saved model and vectorizer
model = joblib.load("model_india.pkl")
vectorizer = joblib.load("vectorizer_india.pkl")

# Same cleaning function we used during training
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]

    label = "Real" if prediction == 1 else "Fake"
    confidence = round(max(probability) * 100, 2)

    return jsonify({"label": label, "confidence": confidence})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)