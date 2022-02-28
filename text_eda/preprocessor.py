"""
Created on Tue Dec 29 14:06:51 2020
Preprocessor script to tokenize, stem, lemmatize, de-noise input
https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html
@author: TNIKOLIC
"""
import streamlit as st
import warnings
import re, unicodedata
import nltk
import contractions
import inflect
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('omw-1.4')
from textblob import TextBlob
from bs4 import BeautifulSoup
import spacy

def remove_html(words):
    # language agnostic
    soup = BeautifulSoup(words, 'lxml')
    clean_words = soup.get_text()
    return clean_words

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = [unicodedata.normalize('NFKD', w).encode('ascii', 'ignore').decode('utf-8', 'ignore') for w in words]
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = [word.lower() for word in words]
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = [re.sub(r'[^\w\s]', '', word) for word in words]

    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()

    lemmas = [lemmatizer.lemmatize(i, pos="a") for i in words]

    lemmas = " ".join(lemmas)

    return lemmas

def normalize(words):
    #de-noising objects
    url_pattern = re.compile(re.compile(r'https?://\S+|www\.S+'))
    email_pattern = re.compile(re.compile(r'\S*@\S*\s?'))

    words = [url_pattern.sub('', w) for w in words]
    words = [email_pattern.sub('', w) for w in words]
    
    words = remove_non_ascii(words)
    
    words = to_lowercase(words)
    
    words = remove_punctuation(words)

    words = replace_numbers(words)
   
    words = remove_stopwords(words)

    return words

# suppress warnings because we are using a progress bar within a 
# cache function
@st.cache(suppress_st_warning=True)
def clean_data(df,feature):
    """
    function to:
        1. de-noise
            a) remove email and other symbols
        2. tokenize
        3. normalize
            a) remove stopwords with NLTK
            b) lemmatize with NLTK
            c) replace numbers with words 
    output: pandas dataframe

    """
    df = df.dropna()

    bar = st.progress(0)

    # use lambda function to run through rows (entries)
    df[feature] = df[feature].apply(lambda x: contractions.fix(x))

    bar.progress(25)

    #st.success("Contractions fixed! For example, you're is turned into you are.")

    df[feature] =  df[feature].apply(lambda x: nltk.word_tokenize(x))

    bar.progress(50)

    #st.success("Tokenized text!")

    df[feature] = df[feature].apply(lambda x: normalize(x))

    bar.progress(75)

    #st.success("Normalized text! Removed stop words, noise, punctuation and turned numbers to text.")

    df[feature] = df[feature].apply(lambda x: lemmatize_verbs(x))

    bar.progress(100)

    #st.success("Lemmatized verbs!")
     
    return df
