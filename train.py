
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle, os
from preprocess import clean_text

def label_from_rating(r):
    if r >=4:
        return 'positive'
    elif r==3:
        return 'neutral'
    else:
        return 'negative'

def main(data_path):
    df = pd.read_excel(data_path)
    df = df.dropna(subset=['review','rating'])
    df['clean_review'] = df['review'].astype(str).apply(clean_text)
    df['label'] = df['rating'].apply(label_from_rating)
    X = df['clean_review']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xte)
    print(classification_report(y_test, preds))
    print("Confusion matrix:\\n", confusion_matrix(y_test, preds))
    os.makedirs('../models', exist_ok=True)
    with open('../models/vectorizer.pkl','wb') as f:
        pickle.dump(vec, f)
    with open('../models/model.pkl','wb') as f:
        pickle.dump(clf, f)
    print('Saved models to /models')

if __name__ == '__main__':
    main('../data/chatgpt_style_reviews_dataset.xlsx')
