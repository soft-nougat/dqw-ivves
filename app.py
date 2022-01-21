# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 17:04:46 2020

Script with defined app, including styling.

@author: TNIKOLIC
"""

import streamlit as st
    
# app setup 
try:

    from helper_functions import *
    
    # set bg
    set_bg_hack('dqw_background.png')
    
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
                                       "Unstructured, audio data"))
    
    if selected_structure == "Structured data":
        
        from tabular_eda.structured_data import *
        structured_data_app()
        
    elif selected_structure == "Unstructured, text data":
        
        from text_eda.text_data import *
        text_data_app()

        
    elif selected_structure == "Unstructured, audio data":
        
        from audio_eda.audio_data import *
        audio_data_app()
    

except KeyError:
    st.error("Please select a key value from the dropdown to continue.")
    
except ValueError:
    st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
    
except TypeError:
    st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
