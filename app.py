import streamlit as st
import pickle
import re
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ===============================
# 1. LOAD MODEL
# ===============================
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# ===============================
# 2. NLP SETUP
# ===============================
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# ===============================
# 3. UI DESIGN
# ===============================
st.set_page_config(page_title="AI Echo", layout="centered")

st.title("AI Echo - Sentiment Analyzer")
st.write("Analyze user reviews and predict whether the sentiment is Positive, Neutral, or Negative.")

st.markdown("---")

# ===============================
# 4. USER INPUT
# ===============================
user_input = st.text_area("Enter your review:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review before predicting.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        result = model.predict(vec)[0]

        st.subheader("Prediction Result:")

        if result == "Positive":
            st.success("😊 Positive Sentiment")
        elif result == "Negative":
            st.error("😠 Negative Sentiment")
        else:
            st.warning("😐 Neutral Sentiment")

st.markdown("---")

# ===============================
# 5. DATA INSIGHTS (EDA)
# ===============================
st.subheader("Dataset Insights")

try:
    df = pd.read_csv("data/final_cleaned_data.csv")

    # Sentiment Distribution
    st.write("Sentiment Distribution:")
    st.bar_chart(df['sentiment'].value_counts())

    # Rating Distribution
    st.write("Rating Distribution:")
    st.bar_chart(df['rating'].value_counts())

except:
    st.info("Dataset not found for visualization.")


