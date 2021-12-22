import json
import streamlit as st
import pandas as pd
import base64
import SessionState
import streamlit.components.v1 as components
from PIL import Image
import SessionState
import pandas as pd

# ----------------------------------------------
# session state
# needs to be refined, session state is used to
# successfully cache objects so the app runs
# smoothly
ss = SessionState.get(is_file_uploaded=False,
                     is_batch_process = False,
                     txt = 'Paste the text to analyze here')

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
            ss.is_batch_process = True
            ss.is_file_uploaded = True
    elif 'json' in data_input_mthd:
        df,ss.txt= get_input(data_input_mthd,
                             ss_text= ss.txt,
                             is_batch=True)
        if df.shape[0]>0:
            ss.is_batch_process = True
            ss.is_file_uploaded = True
        
    return df,ss.txt