import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ===============================
# 1. LOAD DATA
# ===============================
df = pd.read_csv("data/final_cleaned_data.csv")

# Features & Target
X = df['cleaned_review']
y = df['sentiment']

# ===============================
# 2. TEXT → NUMBERS (TF-IDF)
# ===============================
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# ===============================
# 3. TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# ===============================
# 4. TRAIN MODELS
# ===============================

# Logistic Regression
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# ===============================
# 5. EVALUATE MODELS
# ===============================

def evaluate_model(name, model):
    y_pred = model.predict(X_test)
    
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Evaluate both
evaluate_model("Logistic Regression", lr_model)
evaluate_model("Naive Bayes", nb_model)

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, lr_model.predict(X_test))

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===============================
# 6. SAVE BEST MODEL
# ===============================

# Assume Logistic Regression is better (usually is)
pickle.dump(lr_model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("\nModel saved successfully!")



