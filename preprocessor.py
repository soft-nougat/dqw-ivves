# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 14:06:51 2020

@author: TNIKOLIC
"""

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import re
import streamlit as st


# Gensim
import gensim, spacy, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import TfidfModel
warnings.filterwarnings("ignore",category=DeprecationWarning)

stop_words = list(set(stopwords.words('english')))
stop_words.extend(['lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 
'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do',
'done', 'try', 'many','from', 'subject', 're', 'edu','some', 'nice', 'thank',
'think', 'see', 'rather', 'easy', 'easily', 'lot', 'line', 'even', 'also', 'may', 'take', 'come'])

#lemmatization and stemming
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
stemmer = SnowballStemmer(language='english')


url_pattern = re.compile(re.compile(r'https?://\S+|www\.S+'))
email_pattern = re.compile(re.compile(r'\S*@\S*\s?'))

def clean_data(df,feature):
    """
    funtion clean the data by removing unwanted words, patterns and puntuations  
    out: df : pandas dataframe
    
    """
    doc = []
    for entry in df[feature]:
        
        tokens = entry.split()
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [word.lower() for word in tokens if len(word) > 1]
        #tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
        tokens = [w for w in tokens if w not in stop_words]
        tokens = [url_pattern.sub('', w) for w in tokens]
        tokens = [email_pattern.sub('', w) for w in tokens]
        doc.append(' '.join(tokens))                         
    df[feature]= doc 
    return df

def encode(text_list,mode='Term Frequency'):
    """
    funtion encode cleaned data 
    out: id2word, corpus : encoded data 
    
    """
    # Create Dictionary
    id2word = corpora.Dictionary(text_list)

    if mode=='Term Frequency':
        # Create Corpus: Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in text_list]
    else:
        #TF-IDF
        BoW_corpus = [id2word.doc2bow(text) for text in text_list]
        tfidf = models.TfidfModel(BoW_corpus, smartirs='ntc')
        corpus = tfidf[BoW_corpus]
    return id2word, corpus

# ENCODER SECTION #
def extract_features(df,feature,normalization_mthd ='Lemmatization' ,
                     mode='Term Frequency',
                     allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    
    """Form Bigrams, Trigrams and (Lemmatization or stemming)"""

    texts_list = []

    entries = [entry for entry in df[feature]]
    texts_words = [gensim.utils.simple_preprocess(str(text), deacc=True) for text in entries]

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(texts_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[texts_words], threshold=100)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    
    # texts = [word for word in simple_preprocess(str(doc)) if word not in stop_words]
    texts = [bigram_mod[doc] for doc in texts_words]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts_words]
    
    for text in texts:
        if normalization_mthd=='None':
            texts_list.append(text)
        elif normalization_mthd =='Lemmatization':
            doc = nlp(" ".join(text)) 
            
            texts_list.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
           
        else:
            texts_list.append([stemmer.stem(token) for token in text])
            
    
    # remove stopwords once more after lemmatization or stemming
    texts_list = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_list]    
    
    id2word, corpus = encode(texts_list,mode=mode)
    
    return id2word, corpus, texts_list