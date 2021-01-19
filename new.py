import streamlit as st
import pandas as pd
import numpy as np
import urllib as urllib
import matplotlib.pyplot as plt 
import base64
# download english stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import seaborn as sns
from collections import Counter
import gensim
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
import pyLDAvis.gensim
from wordcloud import WordCloud, STOPWORDS

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
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

set_png_as_page_bg('dqw_background.png')

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache
def get_data():
    news = pd.read_csv('abcnews-date-text.csv',nrows=10000)
    return news

try:
    st.header("Data Quality Wrapper")
    news = get_data()
    plots = st.multiselect(
        "Choose 1 plot", ('Length of text', 
                        'Word count',
                        'Average word length',
                        'Stopwords',
                        'Exclude Stopwords',
                        'Topic modelling',
                        'Wordcloud')
    )
    if not plots:
        st.write("Welcome to the DQW for Text analysis. ",
                 "As unstructured data, text input analysis for ",
                 "NLP models is of crucial importance. This dashboard ",
                 "offers visualisation of descriptive statistics of a ",
                 "text input file placed in shared folder. The file name ",
                 "ïs abcnews-date-text.csv. To begin analysis, "
                 "please select at least one plot from the dropdown.")
        
        st.subheader("Preview of input data")
        st.write(news.head(5))
        
        # TO DO: make file name dynamic and set input data cleanup
        # based on file format. Right now, only csv is supported.
    elif 'Length of text' in plots:
        
        st.header("Preliminary Descriptive statistics")
        
        st.write("Number of characters present in each sentence.")
        
        colors = ['purple'] 
          
        plt.hist(news['headline_text'].str.len(), 
                 color = colors) 
        
        plt.xlabel('Character count')
        plt.ylabel('Headline number')
          
        plt.title('Length of text\n\n', 
                  fontweight ="bold") 
  
        st.pyplot()
    elif "Word count" in plots:
         
        st.header("Preliminary Descriptive statistics")
        
        st.write("Number of words present in each headline.")
        
        colors = ['blue'] 
          
        plt.hist(news['headline_text'].str.split().map(lambda x: len(x)), 
                 color = colors) 
        
        plt.xlabel('Word count')
        plt.ylabel('Headline number')
          
        plt.title('Word count\n\n', 
                  fontweight ="bold") 
        
        st.pyplot()
    elif "Average word length" in plots:
        
        st.header("Preliminary Descriptive statistics")
        
        st.write("Average word length, including stopwords.")
        
        colors = ['green'] 
          
        plt.hist(news['headline_text'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)), 
                 color = colors) 
        
        plt.xlabel('Average word length in characters')
        plt.ylabel('Headline number')
          
        plt.title('Average word length\n\n', 
                  fontweight ="bold") 
        
        st.pyplot()
    elif "Stopwords" in plots:
        
        st.header("Stopwords with NLTK")
        
        st.write("Stopword count in headlines.")
        
        # remove stopwords with nltk
        # define stop as nltk stopwords corpus
        stop=nltk.corpus.stopwords.words("english")
    
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
            plt.bar(x,y)
            
            plt.xlabel('Stopword')
            plt.ylabel('Count in all headlines')
          
            plt.title('Stopword count in headlines\n\n', 
                      fontweight ="bold") 
        
        plot_top_stopwords_barchart(news['headline_text'])
        
        st.pyplot()
        
    elif "Exclude Stopwords" in plots:
        
        st.header("Count words, excluding stopwords")
        
        st.write("Word count in headlines.")
        
        # remove stopwords with nltk
        # define stop as nltk stopwords corpus
        stop=nltk.corpus.stopwords.words("english")
    
        def plot_top_non_stopwords_barchart(text):
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
                    
            sns.barplot(x=y,y=x)
            
            plt.xlabel('Count in all headlines')
            plt.ylabel('Word')
          
            plt.title('Word count in headlines\n\n', 
                      fontweight ="bold") 
        
        plot_top_non_stopwords_barchart(news['headline_text'])
        
        st.pyplot()  
    elif "Topic modelling" in plots:
        
        st.header("Topic modelling with LDA")
        
        st.write("On the left side, the area of each circle ",
                 "represents the importance of the topic relative to the ",
                 "corpus. As there are four topics, we have four circles. ",
                 "The distance between the center of the circles indicates ",
                 "the similarity between the topics. ",
                 "On the right side, the histogram of each topic shows the ",
                 "top 30 relevant words.")
    
        def get_lda_objects(text):
            nltk.download('stopwords')    
            stop=set(stopwords.words('english'))
            
            def _preprocess_text(text):
                corpus=[]
                stem=PorterStemmer()
                lem=WordNetLemmatizer()
                for news in text:
                    words=[w for w in word_tokenize(news) if (w not in stop)]
        
                    words=[lem.lemmatize(w) for w in words if len(w)>2]
        
                    corpus.append(words)
                return corpus
            
            corpus=_preprocess_text(text)
            
            dic=gensim.corpora.Dictionary(corpus)
            bow_corpus = [dic.doc2bow(doc) for doc in corpus]
            
            lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                           num_topics = 4, 
                                           id2word = dic,                                    
                                           passes = 10,
                                           workers = 2)
            
            return lda_model, bow_corpus, dic

        def plot_lda_vis(lda_model, bow_corpus, dic):
            #pyLDAvis.enable_notebook()
            vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dic)
            return vis
        
        lda_model, bow_corpus, dic = get_lda_objects(news['headline_text'])
        
        plot_lda_vis(lda_model, bow_corpus, dic)
        
        st.pyplot()  
    elif "Wordcloud" in plots:
        
        st.header("Wordcloud visualisation")
        
        def plot_wordcloud(text):
            nltk.download('stopwords')
            stop=set(stopwords.words('english'))
        
            def _preprocess_text(text):
                corpus=[]
                stem=PorterStemmer()
                lem=WordNetLemmatizer()
                for news in text:
                    words=[w for w in word_tokenize(news) if (w not in stop)]
        
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
    
        plot_wordcloud(news['headline_text'])
        
        st.pyplot()
except urllib.error.URLError as e:
    st.error(
        """
        **This demo requires internet access.**

        Connection error: %s
    """
        % e.reason
    )