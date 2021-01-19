# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 14:40:59 2020

@author: TNIKOLIC
"""
import streamlit as st
import spacy
from collections import  Counter
import seaborn as sns
import matplotlib.pyplot as plt

def plot_named_entity_barchart(text):
    nlp = spacy.load("en_core_web_sm")
    
    def _get_ner(text):
        doc=nlp(text)
        return [X.label_ for X in doc.ents]
    
    ent=text.apply(lambda x : _get_ner(x))
    ent=[x for sub in ent for x in sub]
    counter=Counter(ent)
    count=counter.most_common()
    
    x,y=map(list,zip(*count))
    sns.barplot(x=y,y=x)
    
    plt.xlabel('Row number')
    plt.ylabel('NER')
          
    plt.title('NER instance count\n\n', 
              fontweight ="bold")
    
    st.pyplot()
    
    # get unique list elements
    ent = set(ent)
    
    return ent
    
# visualise tokens per entity
def plot_most_common_named_entity_barchart(text, entity):
    
        
        nlp = spacy.load("en_core_web_sm")
        
        def _get_ner(text,ent):
            doc=nlp(text)
            return [X.text for X in doc.ents if X.label_ == ent]
    
        entity_filtered=text.apply(lambda x: _get_ner(x,entity))
        entity_filtered=[i for x in entity_filtered for i in x]
        
        counter=Counter(entity_filtered)
        x,y=map(list,zip(*counter.most_common()))
        sns.barplot(y,x).set_title(entity)
              
        
        st.pyplot()
    

    
