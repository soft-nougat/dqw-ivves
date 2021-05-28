# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 17:04:46 2020

Script with defined app, including styling.

@author: TNIKOLIC
"""
import json
import plots
import streamlit as st
import pandas as pd
import base64
import SessionState
# Load the LDA model from gensim
import preprocessor as pp
import sweetviz as sv
import streamlit.components.v1 as components

# ----------------------------------------------
# session state
# needs to be refined, session state is used to
# successfully cache objects so the app runs
# smoothly
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


def get_input(data_input_mthd,ss_text,is_batch=False,text_column = "text"):
    """
    function get input from user either by uploading a csv file, pasting text
    or importing json files
    ----------
    ss_text: string
    is_batch: bool 
    text_column: str -> the columnn name for creating pd.DataFrame is _is_batch is False
    """
    if 'CSV' in data_input_mthd:
        uploaded_file = st.file_uploader("Choose a csv file to analyse", type="csv")

        if uploaded_file is not None:
            st.success('File upload successful')
            df = pd.read_csv(uploaded_file)
            return df,ss_text
        else:
            st.info('Please upload a csv file')
            return pd.DataFrame(),ss_text

    elif 'Copy-Paste text' in data_input_mthd: 
        ss_text = st.text_area("Type in text to analyse", ss_text)
        df = pd.DataFrame(data=[ss_text],columns=[text_column])
        return df,ss_text
    
    elif 'json' in data_input_mthd:
        uploaded_file = st.file_uploader("Choose a json file to analyse",
                                         type = "json")

        if uploaded_file is not None:
            st.success('File upload successful')
            data = json.load(uploaded_file)
            df = pd.json_normalize(data)
            #df = pd.read_json(uploaded_file)
            return df,ss_text
        else:
            st.info('Please upload a json file')
            return pd.DataFrame(),ss_text
 
    
def check_input_method(data_input_mthd):
    """
    function check user input method if uploading or pasting or using
    a json file
    Parameters
    ----------
    data_input_mthd: str -> the default displayed text for decision making
    """

    if 'Copy-Paste text' in data_input_mthd:
        df,ss.txt = get_input(data_input_mthd,
                              ss_text= ss.txt)


    elif 'CSV' in data_input_mthd:
        df,ss.txt= get_input(data_input_mthd,
                             ss_text= ss.txt,
                             is_batch=True)
        if df.shape[0]>0:
            # ss.is_batch_process = True
            ss.is_file_uploaded = True
    elif 'json' in data_input_mthd:
        df,ss.txt= get_input(data_input_mthd,
                             ss_text= ss.txt,
                             is_batch=True)
        if df.shape[0]>0:
            # ss.is_batch_process = True
            ss.is_file_uploaded = True
        
    
    return df,ss.txt

def structured_data_app():
    
    st.write("Welcome to the DQW for structured data analysis. ",
                  "Structured data analysis is an important step ",
                  "in AI model development or Data Analysis. This app ",
                  "offers visualisation of descriptive statistics of a ",
                  "csv input file by using the sweetviz package.",
                  " You can pick to analyse only 1 file or compare 2.")
    
    # Side panel setup
    # Step 1 includes Uploading 
    display_app_header(main_txt = "Step 1",
                      sub_txt= "Upload data",
                      is_sidebar=True)
        
    data_input_mthd = st.sidebar.radio("Select Data Input Method",
                                       ('Upload a CSV file',
                                        'Upload a json file'))
    
    selected_structure = st.selectbox("Choose type of analysis", 
                                      ("Analyse 1 file", 
                                       "Compare 2 files"))
    
    
    if selected_structure == "Compare 2 files":
        
        st.subheader('Choose data to analyse :alembic:')
        
        uploaded_files = st.file_uploader("Upload CSVs to compare", 
                                          type="csv", 
                                          accept_multiple_files=True)
        
        data = []
        for file in uploaded_files:
        	dataframe = pd.read_csv(file)
        	file.seek(0)
        	data.append(dataframe)
        
        st.subheader('A preview of input files is below, please wait for data to be compared :bar_chart:')
        st.write(data[0].head(5))
        st.write(data[1].head(5))
        
        my_report = sv.compare([data[0], "Input file 1"], [data[1], "Input file 2"])
        
        my_report.show_html(layout='vertical',
                            open_browser=True)
        
        #profile = ProfileReport(data, title='Your input data profile report').to_html()
        display = open("SWEETVIZ_REPORT.html", 'r', encoding='utf-8')
        source_code = display.read() 
        # display html page in streamlit
        components.html(source_code, height = 600, scrolling=True) 
    
    if selected_structure == "Analyse 1 file":
        
        st.subheader('Choose data to analyse :alembic:')
        data,txt  = check_input_method(data_input_mthd)
        
        st.subheader('A preview of input data is below, please wait for data to be analyzed :bar_chart:')
        st.write(data.head(5))
        
        my_report = sv.analyze(data)
        
        my_report.show_html(layout='vertical',
                            open_browser=True)
        
        #profile = ProfileReport(data, title='Your input data profile report').to_html()
        display = open("SWEETVIZ_REPORT.html", 'r', encoding='utf-8')
        source_code = display.read() 
        # display html page in streamlit
        components.html(source_code, height = 600, scrolling=True) 
    
    
    
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
            ss.df = pp.clean_data(data,feature=text_column)
            st.success('Data cleaning successfuly done')
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
              df_download= pd.DataFrame(ss.df)
              #df_download
              csv = df_download.to_csv(index=False)
              b64 = base64.b64encode(csv.encode()).decode()  # some strings
              linko= f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
              st.markdown(linko, unsafe_allow_html=True)
              
    
    st.subheader('A preview of input data is below, please select plot to start analysis :bar_chart:')
    st.write(ss.df.head(5))
    
    plots.plot(selected_plot,
               ss.df,
               text_column)
    

# app setup 
try:
    
    # set bg
    set_png_as_page_bg('dqw_background.png')
    
    # hide warning for st.pyplot() deprecation
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # Main panel setup
    display_app_header(main_txt='Data Quality Wrapper',
                       sub_txt='Clean, describe, visualise and select data for AI models')
    
    # prompt the user with an option to select which data they want to 
    # analyse
    selected_structure = st.selectbox("Choose data structure to analyse", 
                                      ("Structured data", 
                                       "Unstructured, text data"))
    
    if selected_structure == "Structured data":
        
        structured_data_app()
        
    elif selected_structure == "Unstructured, text data":
        
        text_data_app()
    

except KeyError:
    st.error("Please select a key value from the dropdown to continue.")
    
except ValueError:
    st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
    
except TypeError:
    st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
