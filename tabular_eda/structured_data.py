# this script contans structured data
# analysis code

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from tabular_eda.te import *
import SessionState

# ----------------------------------------------
# session state
# needs to be refined, session state is used to
# successfully cache objects so the app runs
# smoothly
ss = SessionState.get(table_evaluator = None)

def structured_data_app():
    
    st.write("Welcome to the DQW for structured data analysis. ",
            "Structured data analysis is an important step ",
            "in AI model development or Data Analysis. This app ",
            "offers visualisation of descriptive statistics of a ",
            "csv input file by using the pandas profiling package.",
            " You can pick to analyse only 1 file or compare 2.",
            " The app uses the table-evaluator package to compare ",
            "2 tables.")
    
    selected_structure = st.selectbox("Choose type of analysis", 
                                      ("Analyse 1 file", 
                                       "Compare 2 files"))
    
    st.subheader('Choose data to analyse :alembic:')

    if selected_structure == "Compare 2 files":
        

        original = st.file_uploader("Upload reference dataset", 
                                type="csv")

        if original:

            original = pd.read_csv(original)                        

            comparison = st.file_uploader("Upload comparison dataset", 
                                         type="csv") 

            if comparison:                      
            
                comparison = pd.read_csv(comparison)

                st.subheader('A preview of input files is below, please wait for data to be compared :bar_chart:')
                st.subheader('Reference data')
                st.write(original.head(5))
                st.subheader('Comparison data')
                st.write(comparison.head(5))

                ss.table_evaluator = TableEvaluator(original, comparison)
                ss.table_evaluator.visual_evaluation()
                
                dataset_columns = original.columns
                options_columns = dataset_columns.insert(0, 'None')
                
                evaluate_col = st.selectbox('Select a categorical column to analyse:', 
                                            options_columns, 
                                            index=0)

                if evaluate_col != 'None':
                    
                    evaluate = ss.table_evaluator.evaluate(target_col = evaluate_col)

                else:

                    st.warning('Please select a categorical column to analyse.')

        else:

            st.warning("Please upload a reference/original dataset.")
    
    if selected_structure == "Analyse 1 file":
        
        data = st.file_uploader("Upload dataset", 
                                type="csv") 
    
        if data:
            st.subheader('A preview of input data is below, please wait for data to be analyzed :bar_chart:')
            data = pd.read_csv(data)
            st.write(data.head(5))
            
            pr = data.profile_report()
            st_profile_report(pr)