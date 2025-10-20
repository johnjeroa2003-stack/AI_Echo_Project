
import streamlit as st
import pickle, os
from src.preprocess import clean_text

st.title('AI Echo — Sentiment Classifier Demo')
review = st.text_area('Review', height=150, value='This app answers very well and is helpful.')
if st.button('Predict'):
    if not os.path.exists('models/model.pkl'):
        st.error('Model not found. Run `python src/train.py` first to train and save the model.')
    else:
        with open('models/vectorizer.pkl','rb') as f:
            vec = pickle.load(f)
        with open('models/model.pkl','rb') as f:
            clf = pickle.load(f)
        clean = clean_text(review)
        X = vec.transform([clean])
        pred = clf.predict(X)[0]
        st.success(f'Predicted sentiment: {pred}')
        probs = clf.predict_proba(X)[0]
        labels = clf.classes_
        st.write('Confidence:')
        for lab, p in zip(labels, probs):
            st.write(f"- {lab}: {p:.3f}")
