# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 17:04:46 2020

Script with defined app, including styling.

@author: TNIKOLIC
"""

import streamlit as st
from PIL import Image
    
# app setup 
try:

    from helper_functions import *

    # Set website details
    st.set_page_config(page_title ="Data Quality Wrapper", 
                       page_icon='📊', 
                       layout='centered')
    
    # set sidebar width
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 300px;
        margin-left: -300px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

    
    # set bg
    set_bg_hack('dqw_background.png')

    # get logo
    logo = Image.open('logo.png')
    st.sidebar.image(logo, 
                        use_column_width=True)
    
    # hide warning for st.pyplot() deprecation
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # Main panel setup
    display_app_header(main_txt='Data Quality Wrapper',
                       sub_txt='Clean, describe, visualise and select data for AI models')
    
    # prompt the user with an option to select which data they want to 
    # analyse
    selected_structure = st.selectbox("Choose data structure to analyse", 
                                      ("Structured data", 
                                       "Unstructured, text data",
                                       "Unstructured, audio data",
                                       "Unstructured, image data"))
    
    if selected_structure == "Structured data":
        
        from tabular_eda.structured_data import *
        structured_data_app()
        
    elif selected_structure == "Unstructured, text data":
        
        from text_eda.text_data import *
        text_data_app()

    elif selected_structure == "Unstructured, audio data":

        from audio_eda.audio_data import *
        audio_data_app()
    
    elif selected_structure == "Unstructured, image data":
        
        
        from image_eda.image_data import *
        image_data_app()
    

except KeyError:
    st.error("Please select a key value from the dropdown to continue.")
    
except ValueError:
    st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
    
except TypeError:
    st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
