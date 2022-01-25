# this script contans structured data
# analysis code

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from tabular_eda.te import *
from helper_functions import display_app_header

def structured_data_app():
    
    st.write("Welcome to the DQW for structured data analysis. ",
            "Structured data analysis is an important step ",
            "in AI model development or Data Analysis. This app ",
            "offers visualisation of descriptive statistics of a ",
            "csv input file by using the pandas profiling package.",
            " You can pick to analyse only 1 file or compare 2.",
            " The app uses the table-evaluator package to compare ",
            "2 tables.")

    # Side panel setup
    display_app_header(main_txt = "Step 1",
                       sub_txt= "Choose type of analysis",
                       is_sidebar=True)

    selected_structure = st.sidebar.selectbox("", 
                                                ("Analyse 1 file", 
                                                "Compare 2 files"))

    if selected_structure == "Compare 2 files":
        
        display_app_header(main_txt = "Step 2",
                            sub_txt= "Upload data",
                            is_sidebar=True)

        original = st.sidebar.file_uploader("Upload reference dataset", 
                                            type="csv")

        if original:

            original = pd.read_csv(original)                        

            comparison = st.sidebar.file_uploader("Upload comparison dataset", 
                                                  type="csv") 

            if comparison:                      
            
                comparison = pd.read_csv(comparison)

                st.subheader('A preview of input files is below, please wait for data to be compared :bar_chart:')
                st.subheader('Reference data')
                st.write(original.head(5))
                st.subheader('Comparison data')
                st.write(comparison.head(5))

                # ----------------------------------------------
                # session state
                st.session_state['table_evaluator'] = None

                st.session_state.table_evaluator = TableEvaluator(original, comparison)
                st.session_state.table_evaluator.visual_evaluation()
                
                dataset_columns = original.columns
                options_columns = dataset_columns.insert(0, 'None')
                
                evaluate_col = st.selectbox('Select a categorical column to analyse:', 
                                            options_columns, 
                                            index=0)

                if evaluate_col != 'None':
                    
                    evaluate = st.session_state.table_evaluator.evaluate(target_col = evaluate_col)

                else:

                    st.warning('Please select a categorical column to analyse.')

        else:

            st.sidebar.warning("Please upload a reference/original dataset.")
    
    if selected_structure == "Analyse 1 file":

        display_app_header(main_txt = "Step 2",
                            sub_txt= "Upload data",
                            is_sidebar=True)
        
        data = st.sidebar.file_uploader("Upload dataset", 
                                type="csv") 
    
        if data:
            st.subheader('A preview of input data is below, please wait for data to be analyzed :bar_chart:')
            data = pd.read_csv(data)
            st.write(data.head(5))
            
            pr = data.profile_report()
            st_profile_report(pr)