# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 17:04:46 2020

@author: TNIKOLIC
"""
import pos
import ner
import polarity
import streamlit as st
import pandas as pd
import numpy as np
import urllib as urllib
import matplotlib.pyplot as plt 
import base64
import lda
# download english stopwords
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
import seaborn as sns
from collections import Counter
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('punkt')
#import pyLDAvis.gensim
from wordcloud import WordCloud, STOPWORDS
import SessionState
# Load the LDA model from gensim
import gensim
import pyLDAvis.gensim     
import streamlit.components.v1 as components
import preprocessor as pp

 # session state 
ss = SessionState.get(output_df = pd.DataFrame(), 
    df_raw = pd.DataFrame(),
    _model=None,
    text_col='text',
    is_file_uploaded=False,
    id2word = None, 
    corpus= None,
    is_valid_text_feat = False,
    to_clean_data = False,
    to_encode = False,
    to_train = False,
    to_evaluate = False,
    to_visualize = False,
    to_download_report = False,
    df = pd.DataFrame(),
    txt = 'Paste the text to analyze here',
    default_txt = 'Paste the text to analyze here',
    clean_text = None,
    ldamodel = None,
    topics_df = None)

# set background, use base64 to read local file
def get_base64_of_bin_file(bin_file):
    """
    function to read png file 
    ----------
    bin_file: png -> the background image in local folder
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    """
    function to display png as bg
    ----------
    png_file: png -> the background image in local folder
    """
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

# display app header and sidebar
# use HTML code to set div
def display_app_header(main_txt,sub_txt,is_sidebar = False):
    """
    function to display major headers at user interface
    ----------
    main_txt: str -> the major text to be displayed
    sub_txt: str -> the minor text to be displayed 
    is_sidebar: bool -> check if its side panel or major panel
    """

    html_temp = f"""
    <h2 style = "color:#F74369; text_align:center; font-weight: bold;"> {main_txt} </h2>
    <p style = "color:#BB1D3F; text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html = True)
    else: 
        st.markdown(html_temp, unsafe_allow_html = True)
        
def display_app_header_1(sub_txt,is_sidebar = False):
    """
    function to display major headers at user interface
    ----------
    main_txt: str -> the major text to be displayed
    sub_txt: str -> the minor text to be displayed 
    is_sidebar: bool -> check if its side panel or major panel
    """

    html_temp = f"""
    <p style = "color:#BB1D3F; text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html = True)
    else: 
        st.markdown(html_temp, unsafe_allow_html = True)
 
def select_text_feature(df) -> pd.DataFrame :
    """
    this function selects the text feature from the uploaded csv file
    ----------
    df: A pandas Dataframe 
    """
    text_col = st.selectbox('Select the text column',(list(df.columns)))
    
    df = df[text_col]
    df =  pd.DataFrame(df)
        
    return df,text_col


def get_input(ss_text,is_batch=False,text_column = "text"):
    """
    function get input from user either by uploading a csv file of  pasting text
    ----------
    ss_text: string
    is_batch: bool 
    text_column: str -> the columnn name for creating pd.DataFrame is _is_batch is False
    """
    if is_batch:
        uploaded_file = st.file_uploader("Choose a csv file to analyse", type="csv")

        if uploaded_file is not None:
            st.success('File upload successful')
            df = pd.read_csv(uploaded_file)
            return df,ss_text
        else:
            st.info('Please upload a csv file')
            return pd.DataFrame(),ss_text

    else: 
        ss_text = st.text_area("Type in text to analyse", ss_text)
        df = pd.DataFrame(data=[ss_text],columns=[text_column])
        return df,ss_text
 
    
def check_input_method(data_input_mthd):
    """
    function check user input method if uploading or pasting
    Parameters
    ----------
    data_input_mthd: str -> the default displayed text for decision making
    """

    if data_input_mthd=='Copy-Paste text':
        df,ss.txt = get_input(ss_text= ss.txt)


    else:
        df,ss.txt= get_input(is_batch=True,ss_text= ss.txt)
        if df.shape[0]>0:
            # ss.is_batch_process = True
            ss.is_file_uploaded = True
    
    return df,ss.txt

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
                    
            if len(dic) == 0:
                
                st.error("Cannot run this plot because the data has been preprocessed or has no stopwords to extract.")
            
            else:
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
            for word,count in most[:40]:
                if (word not in stop):
                    x.append(word)
                    y.append(count)
                    
            sns.barplot(x=y,y=x,palette = 'crest')
            
            plt.xlabel('Count in input')
            plt.ylabel('Word')
          
            plt.title('Word count in input\n\n', 
                      fontweight ="bold")           

def plot(selected_plot):
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
        
    elif "Exclude" in plots:
        
        st.subheader("Count words, excluding stopwords")
        
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
                 "Please start by selecting the number of topics you wish to generate.")
                 
        topics = st.slider("Please select number of topics", 
                  min_value=1, 
                  max_value=10, 
                  value=None, 
                  step=None, 
                  format=None, 
                  key=None)     
        
        st.write("This dashbard firstly offers the LDA topic visualisation plot.",
                 "On the left side, the area of each circle ",
                 "represents the importance of the topic relative to the ",
                 "corpus. ",
                 "The distance between the center of the circles indicates ",
                 "the similarity between the topics. ",
                 "On the right side, the histogram of each topic shows the ",
                 "top 30 relevant words.")
        
        lda.main_function(data[text_column], topics)
        
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
        ent = ner.plot_named_entity_barchart(data[text_column])
        
        st.write("Secondly, it offers specific entity count in input.")
        
        selected_entity = st.multiselect("Choose entity to quantify", 
                                (list(ent)))
        
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

# app setup 
try:
    
    # set bg
    set_png_as_page_bg('dqw_background.png')
    
    # hide warning for st.pyplot() deprecation
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # Main panel setup
    display_app_header(main_txt='Data Quality Wrapper',
                       sub_txt='Clean, describe, visualise and select data for AI models')
    
    st.write("Welcome to the DQW for Text analysis. ",
                 "As unstructured data, text input analysis for ",
                 "NLP models is of crucial importance. This dashboard ",
                 "offers visualisation of descriptive statistics of a ",
                 "text input file uploaded in form of csv or txt. ",
                 "Please select input method on the left, pick if you wish to ",
                 "preprocess it and select the plot you want to use to analyse it.")
    
    # Side panel setup
    # Step 1 includes Uploading and Preprocessing data (optional)
    display_app_header(main_txt = "Step 1",
                       sub_txt= "Upload data",
                       is_sidebar=True)
    
    data_input_mthd = st.sidebar.radio("Select Data Input Method",
                                       ('Copy-Paste text', 
                                        'Upload a CSV file'))
    
    st.subheader('Choose data to analyse :alembic:')
    data,txt  = check_input_method(data_input_mthd)
    
    data,text_column = select_text_feature(data)
    
    display_app_header_1(sub_txt= "Preprocess data",
                       is_sidebar=True)
    
    clean_data_opt = st.sidebar.radio("Choose wisely",
                                      ('Skip preprocessing', 
                                       'Run preprocessing'))
    
    # clean data #######
    if clean_data_opt=='Skip preprocessing':
            st.subheader('Using Raw data :cut_of_meat:')  #Raw data header
            if ss.text_col != ss.default_txt:
                ss.to_encode = True
    else:
            st.subheader('Using Clean Data :droplet:')  #Clean data header
            ss.df = pp.clean_data(data,feature=text_column)
            st.success('Data cleaning successfuly done')
            ss.to_encode = True
    
    if clean_data_opt=='Run preprocessing':
        # final step
        download=st.button('Click here to download clean data')
        if download:
              df_download= pd.DataFrame(data)
              #df_download
              csv = df_download.to_csv(index=False)
              b64 = base64.b64encode(csv.encode()).decode()  # some strings
              linko= f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
              st.markdown(linko, unsafe_allow_html=True)
    
    
    display_app_header(main_txt = "Step 2",
                       sub_txt= "Analyse data",
                       is_sidebar=True)
    
    selected_plot = st.sidebar.radio(
        "Choose 1 plot", ('Length of text', 
                        'Word count',
                        'Average word length',
                        'Stopwords',
                        'Exclude stopwords',
                        'N-grams',
                        'Topic modelling',
                        'Wordcloud',
                        'Sentiment',
                        'NER',
                        'POS',
                        'Complexity Scores')
        )
    
    
    st.subheader('A preview of input data is below, please select plot to start analysis :bar_chart:')
    st.write(data.head(5))
    
    plot(selected_plot)

except KeyError:
    st.error("Please select a key value from the dropdown to continue.")
    
except ValueError:
    st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
    
except TypeError:
     st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
    
# check internet connection and throw error if none
except urllib.error.URLError as e:
    st.error(
        """
        **This demo requires internet access.**

        Connection error: %s
    """
        % e.reason
    )

