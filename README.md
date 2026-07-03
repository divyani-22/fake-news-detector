# In Fake News Detector — India Edition

> A machine learning web app that detects whether an Indian news headline is **Real** or **Fake** — trained on 56,000+ labeled Indian news articles.

🔗 **Live Demo:** [fake-news-detector-rqnc.onrender.com](https://fake-news-detector-rqnc.onrender.com)

---

## 🕵️ What It Does

Paste any Indian news headline into the app and it will:
- Classify it as **Real** or **Fake**
- Show a **confidence percentage** so you know how certain the model is
- Return results instantly, powered by a trained ML model running in the backend

---

## 🖥️ Screenshot



<img width="786" height="647" alt="Screenshot 2026-07-03 144510" src="https://github.com/user-attachments/assets/e8a581a7-ddd9-4f02-846c-76ca8354b269" />


## 🧠 How It Works

1. **Dataset:** Trained on the [IFND (Indian Fake News Dataset)](https://www.kaggle.com/datasets/sonalgarg174/ifnd-dataset) — 56,714 labeled Indian news headlines spanning politics, COVID-19, elections, violence, and more (2013–2021), verified by Indian fact-checkers like Alt News and Boom Live.

2. **Text Preprocessing:** Headlines are lowercased, stripped of punctuation and numbers, and combined before being fed into the model.

3. **Feature Extraction:** Uses **TF-IDF Vectorization** (top 5,000 features) to convert raw text into numerical representations that capture word importance across the corpus.

4. **Model:** **Logistic Regression** with `class_weight="balanced"` to handle the real/fake class imbalance (~2:1 ratio in the dataset).

5. **Backend:** A **Flask** app loads the saved model and vectorizer, exposes a `/predict` endpoint, and serves the frontend.

6. **Frontend:** A custom-designed "case file" themed HTML/CSS/JS page that sends headlines to the backend and displays the verdict.

---

## 📊 Model Performance

| Metric | Fake News | Real News | Overall |
|---|---|---|---|
| Precision | 95% | 95% | 95% |
| Recall | 89% | 98% | — |
| F1-Score | 92% | 96% | — |
| **Accuracy** | — | — | **94.9%** |

> **Note:** The model performs well on structured, headline-style Indian news text. It may not perform reliably on very short phrases, casual language, or non-Indian news content — this is an intentional design choice, not a bug.

---

## ⚠️ Known Limitation (and Why It's Interesting)

The model is a **pattern-matcher**, not a fact-checker. It learned that certain word patterns tend to appear in fake vs. real Indian headlines. This means:

- It catches **sensational/clickbait-style** fake news very well
- It struggles with **official-sounding** fake news (fake claims written to sound like real press releases), since those share vocabulary with genuine news
- This is a **well-documented real-world challenge** in fake news detection — even large language models struggle here

---





## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.14 |
| ML | scikit-learn (TF-IDF + Logistic Regression) |
| Backend | Flask + Gunicorn |
| Frontend | HTML, CSS, JavaScript |
| Fonts | Google Fonts (Special Elite, Courier Prime) |
| Deployment | Render (free tier) |
| Version Control | Git + GitHub |

---

## 📦 Dataset

**IFND — Indian Fake News Dataset**
- 56,714 labeled Indian news headlines
- Labels: `TRUE` / `Fake`
- Categories: Election, Politics, COVID-19, Violence, Miscellaneous
- Source: [Kaggle](https://www.kaggle.com/datasets/sonalgarg174/ifnd-dataset)
- Fact-checked by: Alt News, Boom Live

---

## 💡 What I Learned

- End-to-end ML pipeline: data loading → cleaning → feature extraction → training → evaluation → deployment
- Why class imbalance matters and how `class_weight="balanced"` handles it
- The real-world limitation of word-pattern-based classifiers vs. true fact-checking
- Building a REST API with Flask and connecting it to a frontend
- Git version control and cloud deployment with Render

---

## 🔮 Future Improvements

- [ ] Add full article body text (not just headlines) for richer signal
- [ ] Try a transformer-based model (BERT/IndicBERT) for better understanding of context
- [ ] Add a toggle to switch between "Indian News" and "Global News" models
- [ ] Show which words most influenced the prediction (model explainability)
- [ ] Add more custom examples across diverse Indian news topics

---

## 👩‍💻 Author

**Divyani Papalkar** 

*Built as a first ML project — from dataset to deployed web app.*

---

> *"The model doesn't know the truth. It knows what truth sounds like — and that's both its power and its limitation."*
