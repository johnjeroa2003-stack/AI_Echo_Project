import pandas as pd

# ===============================
# 1. LOAD DATA
# ===============================
df = pd.read_csv("data/chatgpt_reviews.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# ===============================
# 2. HANDLE MISSING VALUES
# ===============================
print("\nMissing Values:")
print(df.isnull().sum())

# Drop rows where review or rating is missing
df = df.dropna(subset=['review', 'rating'])

# Ensure rating is numeric
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Drop rows where rating conversion failed
df = df.dropna(subset=['rating'])

# Convert rating to integer
df['rating'] = df['rating'].astype(int)

print("\nRating Distribution:")
print(df['rating'].value_counts())

# ===============================
# 3. CREATE SENTIMENT COLUMN
# ===============================
def get_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df['sentiment'] = df['rating'].apply(get_sentiment)

print("\nSample Data with Sentiment:")
print(df[['review', 'rating', 'sentiment']].head())

print("\nSentiment Distribution:")
print(df['sentiment'].value_counts())

# ===============================
# 4. NLP TEXT CLEANING
# ===============================
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download only if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return " ".join(words)

# Apply cleaning
df['cleaned_review'] = df['review'].apply(clean_text)

print("\nCleaned vs Original:")
print(df[['review', 'cleaned_review']].head())

# ===============================
# 5. SAVE FINAL CLEANED DATA
# ===============================
df.to_csv("data/final_cleaned_data.csv", index=False)

print("\nData preprocessing completed successfully!")


