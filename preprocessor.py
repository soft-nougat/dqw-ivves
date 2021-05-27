"""
Created on Tue Dec 29 14:06:51 2020
Preprocessor script to tokenize, stem, lemmatize, de-noise input
https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html
@author: TNIKOLIC
"""
import streamlit as st
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import re, unicodedata
import nltk
import contractions
import inflect
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from textblob import TextBlob
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
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
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    
    return words
    
def clean_data(df,feature):
    """
    function to:
        1. de-noise
            a) remove email and other symbols
        2. tokenize
        3. normalize
            a) remove stopwords
            b) lemmatize
    output: pandas dataframe
    
    """
    df = df.dropna()
    
    #de-noising objects
    url_pattern = re.compile(re.compile(r'https?://\S+|www\.S+'))
    email_pattern = re.compile(re.compile(r'\S*@\S*\s?'))
    
    #loop over the column
    doc = []
    for entry in df[feature]:
        
        #1a)
        textBlb = TextBlob(entry)     # Making our first textblob
        textCorrected = textBlb.correct()   # Correcting the text
        tokens = contractions.fix(str(textCorrected))
        #2
        tokens = nltk.word_tokenize(entry)
        #1a)
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [url_pattern.sub('', w) for w in tokens]
        tokens = [email_pattern.sub('', w) for w in tokens]
        #3a)
        tokens = normalize(tokens)
        #3b)
        lemmas = lemmatize_verbs(tokens)
        doc.append(' '.join(lemmas))   
                      
    df[feature]= doc 
    
    return df