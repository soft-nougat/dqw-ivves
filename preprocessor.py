"""
Created on Tue Dec 29 14:06:51 2020
Preprocessor script to tokenize, stem, lemmatize, de-noise input
https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html
https://towardsdatascience.com/nlp-for-beginners-cleaning-preprocessing-text-data-ae8e306bef0f
@author: TNIKOLIC
"""
import streamlit as st
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import re, unicodedata
import nltk
import contractions
import inflect
from nltk.tokenize import RegexpTokenizer
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from textblob import TextBlob
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('punkt')
from bs4 import BeautifulSoup

def remove_html(words):
    soup = BeautifulSoup(words, 'lxml')
    clean_words = soup.get_text()
    return clean_words

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = [unicodedata.normalize('NFKD', w).encode('ascii', 'ignore').decode('utf-8', 'ignore') for w in words]
    #for word in words:
        #new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        #new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = [word.lower() for word in words]
    #new_words = []
    #for word in words:
        #new_word = word.lower()
        #new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = [re.sub(r'[^\w\s]', '', word) for word in words]
    #new_words = []
    #for word in words:
        #new_word = re.sub(r'[^\w\s]', '', word)
        #if new_word != '':
            #new_words.append(new_word)
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
    new_words = [w for w in words if w not in stopwords.words('english')]
    #new_words = []
    #for word in words:
        #if word not in stopwords.words('english'):
            #new_words.append(word)
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

    # work on splits because the nltk lemmatizer works on words 
    # or rather ngrams
    # splits = words.split()

    # join words together as this is the last step
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
    
    #words = remove_punctuation(words) no need for this since we tokenize with regx
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
            a) remove stopwords with NLTK
            b) lemmatize with NLTK
            c) replace numbers with words 
    output: pandas dataframe
    
    """
    df = df.dropna()
    st.write('0')
    # use lambda function to run through rows (entries)
    df[feature] = df[feature].apply(lambda x: contractions.fix(str(x)))
    st.write('1')
    tokenizer = RegexpTokenizer(r'\w+')
    df[feature] = df[feature].apply(lambda x: tokenizer.tokenize(x)) # nltk word_tokenize too slow for pandas df
    st.write('2')
    df[feature] = df[feature].apply(lambda x: normalize(x))
    st.write('3')
    df[feature] = df[feature].apply(lambda x: lemmatize_verbs(x))
    st.write('4')

    #doc = []
    #for entry in df[feature]:
        
        #1a)
        #textBlb = TextBlob(entry)     # Making our first textblob
        #textCorrected = textBlb.correct()   # Correcting the text
        
        #tokens = contractions.fix(str(textCorrected))
        #2
        #tokens = nltk.word_tokenize(tokens)
        #1a)
        
        #tokens = normalize(tokens)
        #3b)
        #lemmas = lemmatize_verbs(tokens)
        #doc.append(' '.join(lemmas))   
                      
    #df[feature]= doc 
    
    return df