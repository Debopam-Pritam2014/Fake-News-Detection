import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pickle
import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
ps = PorterStemmer()
nltk.download('stopwords')
stpwords = stopwords.words('english')

model = pickle.load(open('model1.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))


def transform_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub('[^a-zA-Z]', ' ', text)  # removing commas, question mark digit etc.
    text = text.lower()
    #     text=text.split()
    text = nltk.word_tokenize(text)
    stem_word = [ps.stem(word) for word in text if not word in stpwords]
    stem_word = ' '.join(stem_word)
    return stem_word


st.title('Fake News Detection')
inp = st.text_area('Enter News or title')
if inp:
    trans_inp = transform_text(inp)
    trans_inp=[trans_inp]
    inp_vector = tfidf.transform(trans_inp)
btn = st.button('Click here to Predict')
if btn:
    pred = model.predict(inp_vector)
    if pred==0:
        st.write('Real News')
    else:
        st.write('Fake News Detected')