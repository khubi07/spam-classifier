import streamlit as st
import pickle
import string
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    #1. converted to lower
    text = text.lower()

    #2. tokenize the words and return list
    text = nltk.word_tokenize(text) 
    # we can run a loop on list to remove special char

    #3. removed special char
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    #4. removed stopwords and punctuation
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    #5. stemming
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)
    


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS classifier")
input_sms = st.text_area("Enter the message")
if st.button('Predict'):


    # now we'll hv to work in 4 processes
    # 1. preprocess 
    transformed_text = transform_text(input_sms)

    # 2. vectorize. 
    vector_input = tfidf.transform([transformed_text])

    # 3predict.
    result = model.predict(vector_input)[0]

    #  4. display
    if result == 1:
        st.header("spam")
    else:
        st.header("not spam")
