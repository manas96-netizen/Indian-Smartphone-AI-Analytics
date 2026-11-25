import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Indian Smartphone AI & Market Analyzer", page_icon="📱", layout="wide")

st.title("📱 Indian Smartphone Market & AI Sentiment Analyzer")
st.markdown("End-to-end **Data Analytics + NLP Machine Learning** project analyzing 1,000 Indian smartphone customer reviews.")

# Load data & trained AI model
df = pd.read_csv("ecommerce_1000_reviews.csv")
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Section 1: Live AI Sentiment Tester
st.subheader("🤖 Live AI Review Sentiment Predictor")
user_review = st.text_input("Type or paste any smartphone customer review below:", "Battery drains very fast and phone heats up")

if st.button("Analyze Review Sentiment"):
    vec = vectorizer.transform([user_review])
    pred = model.predict(vec)
    prob = model.predict_proba(vec).max() * 100
    if pred == 1:
        st.success(f"**POSITIVE FEEDBACK 😊** (AI Confidence: {prob:.1f}%)")
    else:
        st.error(f"**NEGATIVE / COMPLAINT ⚠️** (AI Confidence: {prob:.1f}%)")

st.divider()

# Section 2: Data Analyst Brand Insights
st.subheader("📊 Brand Performance & Pricing Insights (n = 1,000)")
col1, col2 = st.columns(2)

brand_stats = df.groupby("Brand").agg(
    Avg_Price_INR=("Price_Clean", "mean"),
    Avg_Rating=("Rating", "mean"),
    Positive_Pct=("Sentiment", lambda x: round(x.mean() * 100, 1))
).round(2)

with col1:
    st.write("**Brand Summary Table**")
    st.dataframe(brand_stats, use_container_width=True)

with col2:
    st.write("**Average Customer Rating by Brand**")
    st.bar_chart(brand_stats["Avg_Rating"])
