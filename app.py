import pickle
import streamlit as st
import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


def text_pp(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf=pickle.load(open('Vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email Spam Classifier")
sms = st.text_input("Enter the sms")

if st.button("Predict"):
    #1.preprocess
    transformed_sms= text_pp(sms)
    #2.vectorisation
    vectored_sms=tfidf.transform([transformed_sms])
    #3.predict
    prediction = model.predict(vectored_sms)
    #4.display
    if prediction == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")