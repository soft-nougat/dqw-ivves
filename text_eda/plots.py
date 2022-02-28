# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:56:29 2021

This scripts holds all code for plots in the DQW app.
It also holds plot descriptions in the app.

@author: TNIKOLIC
"""
import text_eda.pos as pos
import text_eda.ner as ner
import text_eda.polarity as polarity
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt 
import text_eda.lda as lda
# download english stopwords
import nltk
from nltk.corpus import stopwords
import seaborn as sns
from collections import Counter
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('punkt')
#import pyLDAvis.gensim
from wordcloud import WordCloud, STOPWORDS
# Load the LDA model from gensim

def plot_top_ngrams_barchart(text, n=2):
    
    nltk.download('stopwords')
    stop=set(stopwords.words('english'))
        
    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]
        
    def _get_top_ngram(corpus, n=None):
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) 
                             for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:10]
        
    top_n_bigrams=_get_top_ngram(text,n)[:10]
    x,y=map(list,zip(*top_n_bigrams))
    sns.barplot(x=y,y=x, palette = 'mako')

            
 
def plot_wordcloud(text):
    nltk.download('stopwords')
    stop=set(stopwords.words('english'))

    def _preprocess_text(text):
        corpus=[]
        stem=PorterStemmer()
        lem=WordNetLemmatizer()
        for data in text:
            words=[w for w in word_tokenize(data) if (w not in stop)]

            words=[lem.lemmatize(w) for w in words if len(w)>2]

            corpus.append(words)
        return corpus
    
    corpus=_preprocess_text(text)
    
    wordcloud = WordCloud(
        background_color='white',
        stopwords=set(STOPWORDS),
        max_words=100,
        max_font_size=30, 
        scale=3,
        random_state=1)
    
    wordcloud=wordcloud.generate(str(corpus))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
 
    plt.imshow(wordcloud)
    plt.show()
            
def plot_top_stopwords_barchart(text):
    stop=set(stopwords.words('english'))
    
    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]
    from collections import defaultdict
    dic=defaultdict(int)
    for word in corpus:
        if word in stop:
            dic[word]+=1
            
    top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
    x,y=zip(*top)
    plt.bar(x,y,color='c')

    plt.xlabel('Count in input')
    plt.ylabel('Word')
  
    plt.title('Word count in input\n\n', 
              fontweight ="bold")          
    
    st.pyplot()
            
def plot_top_non_stopwords_barchart(text):
    nltk.download('stopwords')
    stop=set(stopwords.words('english'))
    
    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    counter=Counter(corpus)
    most=counter.most_common()
    x, y=[], []
    for word,count in most[:30]:
        if (word not in stop):
            x.append(word)
            y.append(count)
            
    sns.barplot(x=y,y=x,palette = 'crest')
    
    plt.xlabel('Count in input')
    plt.ylabel('Word')
  
    plt.title('Word count in input\n\n', 
              fontweight ="bold")           

def plot(selected_plot,
         data, 
         text_column):
    plots = selected_plot
    if 'Length of text' in plots:
        
        st.subheader("Preliminary Descriptive statistics")
        
        st.write("Number of characters present in each sentence.")
        
        colors = ['purple'] 
          
        plt.hist(data[text_column].str.len(), 
                 color = colors) 
        
        plt.xlabel('Character count')
        plt.ylabel('Row number')
          
        plt.title('Length of text\n\n', 
                  fontweight ="bold") 
  
        st.pyplot()
    elif "Word count" in plots:
         
        st.subheader("Preliminary Descriptive statistics")
        
        st.write("Number of words present in input.")
        
        colors = ['blue'] 
          
        plt.hist(data[text_column].str.split().map(lambda x: len(x)), 
                 color = colors) 
        
        plt.xlabel('Word count')
        plt.ylabel('Row number')
          
        plt.title('Word count\n\n', 
                  fontweight ="bold") 
        
        st.pyplot()
    elif "Average word length" in plots:
        
        st.subheader("Preliminary Descriptive statistics")
        
        st.write("Average word length, including stopwords.")
        
        colors = ['green'] 
          
        plt.hist(data[text_column].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)), 
                 color = colors) 
        
        plt.xlabel('Average word length in characters')
        plt.ylabel('Row number')
          
        plt.title('Average word length\n\n', 
                  fontweight ="bold") 
        
        st.pyplot()
    elif "Stopwords" in plots:
        
        st.subheader("Stopword extraction with NLTK")
        
        st.write("Stopword count in input.")
        
        plot_top_stopwords_barchart(data[text_column])
        
    elif "Unique word count" in plots:
        
        st.subheader("Count unique words, excluding stopwords")
        
        st.write("Word count in input.")
        
        plot_top_non_stopwords_barchart(data[text_column])
        
        st.pyplot()  
        
    elif "N-grams" in plots:
        
        st.subheader("N-gram analysis")
        
        st.write("An n-gram is a contiguous sequence of n items ",
                 "from a given sample of text or speech. The items ",
                 "can be phonemes, syllables, letters, words or base ",
                 "pairs according to the application. This plot quantifies "
                 "2- and 3-grams in the input.")
        
        plot_top_ngrams_barchart(data[text_column],2)
        
        plt.xlabel('Count in input')
        plt.ylabel('2-gram')
          
        plt.title('2-gram count in input\n\n', 
                      fontweight ="bold") 
        
        st.pyplot()
        
        plot_top_ngrams_barchart(data[text_column],3)
        
        plt.xlabel('Count in input')
        plt.ylabel('3-gram')
          
        plt.title('3-gram count in input\n\n', 
                      fontweight ="bold")
        
        st.pyplot()
        
    elif "Topic modelling" in plots:
        
        st.subheader("Topic modelling with LDA")
        
        st.write("Latent Dirichlet Allocation is a  is a generative statistical ",
                 "model that allows sets of observations to be explained by ",
                 "unobserved groups that explain why some parts of the data are similar. ",
                 "Please start either by running model optimization or by selecting the ",
                 "number of topics you wish to generate.")
        
        option = st.multiselect('Would you like to find an optimal number of topics according to your input?',
                               ('Yes', 'No'))

        if option:
        
            if 'Yes' in option:
                
                topics = lda.main_function_optimal(data[text_column])
                
                
            elif 'No' in option:
                
                topics = st.number_input('Select number of topics to generate',
                                        min_value = 1, format = '%i')
                    
                topics = int(topics)
                
                lda.main_function(data[text_column], topics)
            
            st.write("This dashbard firstly offers the LDA topic visualisation plot.",
                    "On the left side, the area of each circle ",
                    "represents the importance of the topic relative to the ",
                    "corpus. ",
                    "The distance between the center of the circles indicates ",
                    "the similarity between the topics. ",
                    "On the right side, the histogram of each topic shows the ",
                    "top 30 relevant words.")
        

            st.write("Secondly, the dashboard displays word importance in each topic.")
            
            lda.word_importance(topics)
        
        
    elif "Wordcloud" in plots:
        
        st.subheader("Wordcloud visualisation")
        
        plot_wordcloud(data[text_column])
        
        st.pyplot()
        
    elif "Sentiment" in plots:
        
        st.subheader("Sentiment analysis with TextBlob & Vader")
        
        st.write("Sentiment analysis is the use of natural language processing, ",
                     "text analysis, computational linguistics, and biometrics to ",
                     "systematically identify, extract, quantify, and study affective ",
                     "states and subjective information.")
        
        st.write("This dashboard firstly offers a polarity score for entire dataset")
        
        polarity.plot_polarity_histogram(data[text_column])
        
        st.pyplot()
        
        st.write("Secondly, it offers sentiment analysis with TextBlob")
        polarity.plot_sentiment_barchart(data[text_column],
                                'TextBlob')
        st.pyplot()
        
        st.write("Thirdly, it offers sentiment analysis with Vader")
        polarity.plot_sentiment_barchart(data[text_column],
                                'Vader')
        st.pyplot()
        
    elif "NER" in plots:
        
        st.subheader("Named Entity Recognition")
        
        st.write("Named-entity recognition (NER) is a subtask of ",
                 "information extraction that seeks to locate and ",
                 "classify named entities mentioned in unstructured ",
                 "text into pre-defined categories such as person names, ",
                 "organizations, locations, medical codes, time expressions, ",
                 "quantities, monetary values, percentages, etc.")
        
        st.write("This dashboard firstly offers NER instance count in input.")
        ent, x, y = ner.plot_named_entity_barchart(data[text_column])
        
        sns.barplot(x=y,y=x)
        plt.xlabel('Row number')
        plt.ylabel('NER')
            
        plt.title('NER instance count\n\n', 
                fontweight ="bold")

        # show plot in streamlit
        st.pyplot()
        
        st.write("Secondly, it offers specific entity count in input.")
        
        selected_entity = st.multiselect("Choose entity to quantify", 
                                (list(ent)))

        if selected_entity:
        
            ner.plot_most_common_named_entity_barchart(data[text_column], entity =
                                                    selected_entity[0])
        
    elif "POS" in plots:
        
        st.subheader("Part of Speech Tagging")
        st.write("A Part-Of-Speech Tagger (POS Tagger) is a piece of ",
                 "software that reads text in some language and assigns ",
                 "parts of speech to each word (and other token), such as noun, ",
                 "verb, adjective, etc.")
        
        st.write("This dashbard firstly offers a count of all POS tags in input.")
        
        pos_1 = pos.plot_parts_of_speach_barchart(data[text_column])
        
        st.write("Secondly, it offers a count of a specific POS tag.")
        
        selected_pos = st.multiselect("Choose tag to quantify", 
                                (list(pos_1)))

        if selected_pos:
        
            pos.plot_most_common_part_of_speech_barchart(data[text_column],
                                                    selected_pos[0])
    
    elif 'Complexity' in plots:
        
        st.subheader("Complexity score in input")
        st.write("In linguistics, complexity is a characteristic of a ",
                 "text but there are multiple measures and hence multiple ",
                 "implied definitions in practice. ",
                 "In natural language processing, these measures are useful ",
                 "for descriptive statistics.")
        
        st.write("This dashboard firstly offers a quantification ",
                 "of Flesch Reading Ease ",
                 "which shows the readibility of text.",
                 " If the score is 90-100, then the text is very easy ",
                 "to read, 80-89 easy, 70-79 fairly easy, 60-69 standard, ",
                 "50-59 fairly difficult, 30-49 difficult and 0-29 very ",
                 "confusing. If the score is negative, it means it's very confusing.")
        
        from textstat import flesch_reading_ease

        data[text_column].\
            apply(lambda x : flesch_reading_ease(x)).\
            hist()
            
        st.pyplot()
