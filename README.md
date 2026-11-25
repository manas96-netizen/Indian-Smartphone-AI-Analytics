# 📱 Indian Smartphone Market & AI Sentiment Analyzer

An end-to-end **Data Analytics & Natural Language Processing (NLP)** web application built to analyze customer satisfaction across 1,000 Indian smartphone reviews and predict review sentiment in real time.

### 🔗 Live Interactive Demo
👉 **[Click here to test the Live Web App](https://streamlit.app)**

### 🛠️ Tech Stack & Methods
* **Data Cleaning & Analysis:** Python (`Pandas`, `NumPy`), Exploratory Data Analysis (EDA)
* **Machine Learning / NLP:** `Scikit-Learn` (`TfidfVectorizer` with bigrams `ngram_range=(1,2)` + `LogisticRegression`)
* **Deployment:** `Streamlit` Community Cloud & GitHub

### 📈 Key Business & Technical Highlights
* **Brand Insights:** Identified that mid-range devices with thermal and battery complaints averaged **3.98/5**, whereas performance-focused models averaged **4.55/5**.
* **Error Analysis & Optimization:** Resolved false-positive sentiment predictions on mixed-keyword reviews (e.g., *"battery drains while gaming"*) by upgrading feature extraction from unigrams to bigrams.

### 📂 Repository Structure
* `app.py` — Streamlit interactive dashboard and live sentiment inference UI
* `ecommerce_1000_reviews.csv` — Cleaned dataset of 1,000 Indian smartphone listings and reviews
* `sentiment_model.pkl` & `tfidf_vectorizer.pkl` — Trained NLP model and bigram vectorizer
* `requirements.txt` — Python dependencies for cloud deployment
