# this script contains code for
# unstructured, text data analysis

import json
import plots
import streamlit as st
import pandas as pd
import preprocessor as pp
import streamlit.components.v1 as components
from PIL import Image
from helper_functions import *
import SessionState

# ----------------------------------------------
# session state
# needs to be refined, session state is used to
# successfully cache objects so the app runs
# smoothly
ss = SessionState.get(selected_structure = None,
                     output_df = pd.DataFrame(), 
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

def text_data_app():
    
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
                                        'Upload a CSV file',
                                        'Import a json file'))
    
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
            
            display_app_header(main_txt = "Step 2",
                   sub_txt= "Analyse data",
                   is_sidebar=True)
            
            selected_plot = st.sidebar.radio(
            "Choose 1 plot", ('Length of text', 
                            'Word count',
                            'Average word length',
                            'Stopwords',
                            'Unique word count',
                            'N-grams',
                            'Topic modelling',
                            'Wordcloud',
                            'Sentiment',
                            'NER',
                            'POS',
                            'Complexity Scores')
            )
            
            if ss.text_col != ss.default_txt:
               ss.to_encode = True
    else:
            st.subheader('Using Clean Data :droplet:')  #Clean data header
            data = pp.clean_data(data,feature=text_column)
            st.success('Data cleaning successfuly done!')
            
            image = Image.open('pp.png')
            st.image(image, caption='Preprocessing steps done by DQW')
            
            ss.to_encode = True
    
    if clean_data_opt=='Run preprocessing':
        display_app_header(main_txt = "Step 2",
                   sub_txt= "Analyse data",
                   is_sidebar=True)
            
        selected_plot = st.sidebar.radio(
        "Choose 1 plot", ('Length of text', 
                        'Word count',
                        'Average word length',
                        'Unique word count',
                        'N-grams',
                        'Topic modelling',
                        'Wordcloud',
                        'Sentiment',
                        'NER',
                        'POS',
                        'Complexity Scores')
        )
        # final step
        download=st.button('Click here to download clean data')
        if download:
              df_download= pd.DataFrame(data)
              #df_download
              csv = df_download.to_csv(index=False)
              b64 = base64.b64encode(csv.encode()).decode()  # some strings
              linko= f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
              st.markdown(linko, unsafe_allow_html=True)
              
    
    st.subheader('A preview of input data is below, please select plot to start analysis :bar_chart:')
    st.write(data.head(5))
    
    plots.plot(selected_plot,
               data,
               text_column)