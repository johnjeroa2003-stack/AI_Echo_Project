import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/final_cleaned_data.csv")

# For time-based charts
df['date'] = pd.to_datetime(df['date'], errors='coerce')

sns.set(style="whitegrid")

#-----Rating-----
plt.figure()
sns.countplot(x='rating', data=df)
plt.title("Rating Distribution (1–5 Stars)")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

#-----Sentiment Distribution-----
plt.figure()
sns.countplot(x='sentiment', data=df)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

#-----Helpful Reviews-----
helpful = df[df['helpful_votes'] > 10]

plt.figure()
sns.countplot(x='sentiment', data=helpful)
plt.title("Helpful Reviews (votes > 10) by Sentiment")
plt.show()

#-----Word Clouds-----
from wordcloud import WordCloud

# Positive
pos_text = " ".join(df[df['sentiment']=="Positive"]['cleaned_review'])
wc_pos = WordCloud(width=800, height=400).generate(pos_text)

plt.figure()
plt.imshow(wc_pos)
plt.axis('off')
plt.title("Positive Reviews WordCloud")
plt.show()

# Negative
neg_text = " ".join(df[df['sentiment']=="Negative"]['cleaned_review'])
wc_neg = WordCloud(width=800, height=400).generate(neg_text)

plt.figure()
plt.imshow(wc_neg)
plt.axis('off')
plt.title("Negative Reviews WordCloud")
plt.show()

#-----Rating Over Time-----
df_time = df.dropna(subset=['date'])

plt.figure()
df_time.groupby(df_time['date'].dt.to_period("M"))['rating'].mean().plot()
plt.title("Average Rating Over Time")
plt.xlabel("Month")
plt.ylabel("Average Rating")
plt.show()

#-----Ratings by Location-----
top_locations = df['location'].value_counts().head(10).index

plt.figure()
sns.boxplot(x='location', y='rating', data=df[df['location'].isin(top_locations)])
plt.xticks(rotation=45)
plt.title("Ratings by Top Locations")
plt.show()

#-----Platform Comparison-----
plt.figure()
sns.barplot(x='platform', y='rating', data=df)
plt.title("Average Rating by Platform")
plt.show()

#-----Verified vs Non-Verified-----
plt.figure()
sns.barplot(x='verified_purchase', y='rating', data=df)
plt.title("Verified vs Non-Verified Users")
plt.show()

#-----Review Length Analysis-----
plt.figure()
sns.boxplot(x='sentiment', y='review_length', data=df)
plt.title("Review Length vs Sentiment")
plt.show()

#-----Best Version-----
plt.figure()
sns.barplot(x='version', y='rating', data=df)
plt.xticks(rotation=45)
plt.title("Average Rating by Version")
plt.show()

