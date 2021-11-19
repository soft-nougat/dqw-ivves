# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 17:04:46 2020

Script with defined app, including styling.

@author: TNIKOLIC
"""

import streamlit as st
import SessionState
import pandas as pd

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
                     topics_df = None,
                     table_evaluator = None)
    
# app setup 
try:

    from helper_functions import *
    
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
        
        from structured_data import *
        structured_data_app()
        
    elif selected_structure == "Unstructured, text data":
        
        from text_data import *
        text_data_app()
    

except KeyError:
    st.error("Please select a key value from the dropdown to continue.")
    
except ValueError:
    st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
    
except TypeError:
    st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
