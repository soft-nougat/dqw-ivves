# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 17:23:54 2020

@author: TNIKOLIC
"""
import streamlit as st
# components allow for pyLDAvis interactive graph display
import streamlit.components.v1 as components
import spacy
spacy.load("en_core_web_sm")
from spacy.lang.en import English
parser = English()
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))
from gensim import corpora
import random
import pyLDAvis.gensim
import gensim
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens   
 
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

def main_function(input, NUM_TOPICS):
    
    text_data = []
    for i in input:
        tokens = prepare_text_for_lda(i)
        if random.random() > .99:
                #print(tokens)
                text_data.append(tokens)
                
    if len(text_data) == 0:
        text_data = [i.split() for i in input]
    
    
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, 
                                               num_topics = NUM_TOPICS, 
                                               id2word=dictionary, 
                                               passes=15)
    #ldamodel.save('model.gensim')
    topics = ldamodel.print_topics(num_words=4)
    for topic in topics:
        st.write(topic) 
        
    ldamodel.save('model.gensim')

    # check if user input is 1 topic, if so, display error
    if NUM_TOPICS == 1:
        
        st.error("Please select more than 1 topic to visualise!")
        
    else:
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, 
                                                   corpus, 
                                                   dictionary, 
                                                   sort_topics=False,
                                                   mds='mmds')
        pyLDAvis.save_html(lda_display, 'lda_display.html')
        
        # display html page in streamlit
        # open file and decode it, then serve
        lda_display = open("lda_display.html", 'r', encoding='utf-8')
        source_code = lda_display.read() 
        components.html(source_code, height = 800, scrolling=True)    
        
    return ldamodel, corpus, dictionary
      
def get_coherence(ldamodel,
                  corpus,
                  dictionary):
    
    from gensim.models import CoherenceModel

        
    coherence_model_lda = CoherenceModel(model=ldamodel, 
                                         corpus=corpus, 
                                         dictionary=dictionary, 
                                         coherence='u_mass')
    
    coherence_lda = coherence_model_lda.get_coherence()
    
    st.write("The u_mass coherence score for this model and number of topics is ", coherence_lda)
    
def word_importance(topics):        
    warnings.filterwarnings("ignore")
    lda = gensim.models.LdaModel.load('model.gensim')
    fiz=plt.figure(figsize=(15,30))
    for i in range(topics):
        df=pd.DataFrame(lda.show_topic(i), columns=['Term','Prob']).set_index('Term')
        
        if topics < 5:
            plt.subplot(5,2,i+1)
        else:
            plt.subplot(topics,2,i+1)
            
        plt.title('Topic '+str(i+1))
        sns.barplot(x='Prob', 
                    y=df.index, 
                    data=df, 
                    label='Cities', 
                    palette='mako')
        
        plt.xlabel('Probability')
    
    st.pyplot()