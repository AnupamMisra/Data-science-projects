import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
items=[]
url=[]



def page_reader(url):

    st.write(url)

    page_request = requests.get(url)
    data = page_request.content
    soup = BeautifulSoup(data,"html.parser")
    title=soup.title.text

    for article in soup.find_all(class_='eachStory'):
        head=article.h3

        if head is not None:
            #print(head)
            if head.find('a') is None:
                head=str(head)[13:-5]
                items.append(head)
    return items   

def wordcl(items):
    sents=[]
    news=''
    ctr=0

    sents=[sent for sent in items]
    for j in range(len(sents)):
        news=news+str(sents[j])
    wowwords=WordCloud(stopwords=ENGLISH_STOP_WORDS.union(['Sensex','Nifty','stock','market','stocks','BSE','NSE','Nifty50','trade','trading','index']), background_color='white').generate(news)

    plt.title('Wordcloud for today\'s news')
    plt.imshow(wowwords,interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

def predict(items):
    sentiments=pd.Series([])
    for i in range(len(items)):
        t=TextBlob(items.iloc[i])
        sentiments[i]=t.sentiment[0]*(1-t.sentiment[1])*len(items.iloc[i])
    sentiments=sentiments[sentiments!=0]
    sentiments=1/(1+np.exp(-sentiments))
    return sentiments.mean()

st.title("Welcome to the news prediction app")
st.markdown("But don't hold me against it!")
st.markdown("If the blog is more than one page long, enter the page URLs as strings in a list")
urls=st.text_input("Please enter the URL(s) for ET live blog")
state=st.button("Get prediction")




if state:
    #for url in urls:
    items.extend(page_reader(urls))
    items=pd.Series(items)
    wordcl(items)
    bullish=predict(items)

    st.write("Percentage change of the market going up tomorrow: ", round(bullish*100), "%")
