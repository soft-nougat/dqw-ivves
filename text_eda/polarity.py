# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 12:16:02 2020

@author: TNIKOLIC
"""

from textblob import TextBlob
    
def plot_polarity_histogram(text):
    
    def _polarity(text):
        return TextBlob(text).sentiment.polarity
        
    polarity_score =text.apply(lambda x : _polarity(x))
    polarity_score.hist(color = 'c')
    
    plt.xlabel('Polarity score')
    plt.ylabel('Row number')
          
    plt.title('Sentiment analysis\n\n', 
              fontweight ="bold")
    
from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

def sentiment_vader(text, sid):
    ss = sid.polarity_scores(text)
    ss.pop('compound')
    return max(ss, key=ss.get)

def sentiment_textblob(text):
        x = TextBlob(text).sentiment.polarity
        
        if x<0:
            return 'Negative'
        elif x==0:
            return 'Neutral'
        else:
            return 'Positive'

def plot_sentiment_barchart(text, method='TextBlob'):
    if method == 'TextBlob':
        sentiment = text.map(lambda x: sentiment_textblob(x))
    elif method == 'Vader':
        nltk.download('vader_lexicon')
        sid = SentimentIntensityAnalyzer()
        sentiment = text.map(lambda x: sentiment_vader(x, sid=sid))
    else:
        raise ValueError('Textblob or Vader')
    
    plt.bar(sentiment.value_counts().index,
            sentiment.value_counts(), color = 'c')
    
    plt.xlabel('Sentiment')
    plt.ylabel('Row number')
          
    plt.title('Sentiment analysis\n\n', 
              fontweight ="bold")

