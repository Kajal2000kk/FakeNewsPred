import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import streamlit as st

# loading the trained model
# #
loaded_model = pickle.load(open('FP.pkl', 'rb'))

#
#
# # creating a function for prediction
#svm = pickle.load(pickle_in)
port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content







def main():
    # front end elements of the web page

    html_temp = """ 
    <div style ="background-color: #2b3845   ;padding:13px"> 
    <h1 style ="font-family:monospace;font-size: 30px;color:#f2eef6;text-align:center;">Fake News Prediction(PDP)</h1> 
    </div> 
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)
    st.sidebar.subheader('Visualization Settings')
    uploaded_file=st.sidebar.file_uploader(label="Upload your csv file or Excel file",type=['csv','xlsx'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)


    subheader= """ <p style="font-family:sans-serif; color: #40E0D0; font-size: 15px;">  Machine Learning Web App to predict fake news,Built with Streamlit, Deployed using Heroku. </p>"""
    st.markdown(subheader, unsafe_allow_html=True)
    text = st.text_input("text")
    prediction = stemming(text)








    if st.button("Predict"):
        result = stemming(text)

        if (result == 0):
            st.success('Real news')
        else:
            st.success('fake news')






if __name__ == '__main__':
    main()


