"""
A script with th strucutured data analysis logic
Additional scripts: report_generation
"""

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from tabular_eda.te import *
from helper_functions import display_app_header, generate_zip_structured
from tabular_eda.report_generation import create_pdf_html, create_pdf

def structured_data_app():

    # ----------------------------------------------
    # session state
    if 'pr' not in st.session_state:
        st.session_state['pr'] = None
    
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

    display_app_header(main_txt = "Step 2",
                    sub_txt= "Upload data",
                    is_sidebar=True)

    if selected_structure == "Compare 2 files":
        
        compare_files()
    
    if selected_structure == "Analyse 1 file":
        
        analyse_file()
            

def compare_files():

    """
    The portion of structured data app dedicated to file comparison
    """

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

            # Side panel setup
            display_app_header(main_txt = "Step 3",
                            sub_txt= "Choose table-evaluator method",
                            is_sidebar=True)

            selected_method = st.sidebar.selectbox("", 
                                                    ("Plot the differences", 
                                                    "Compare categorical columns"))

            if selected_method == "Plot the differences":

                table_evaluator = TableEvaluator(original, comparison)
                table_evaluator.visual_evaluation()

                # Side panel setup
                display_app_header(main_txt = "Step 4",
                                sub_txt= "Download pdf report",
                                is_sidebar=True)

                with st.spinner("The pdf is being generated..."):
                    create_pdf(original, comparison)
                st.success('Done! Please refer to sidebar, Step 4 for download.')

                zip = generate_zip_structured(original, comparison)

                with open("pdf_files/report_files_dqw.zip", "rb") as fp:
                    st.sidebar.download_button(
                            "⬇️",
                        data=fp,
                        file_name="compare_files_dqw.zip",
                        mime="application/zip"
                    )

            else:
                
                # additional analysis part -------
                # insert an additional None column to options to stop the app 
                # from running on a wrong column
                dataset_columns = original.columns
                options_columns = dataset_columns.insert(0, 'None')
                
                evaluate_col = st.selectbox('Select a categorical column to analyse:', 
                                            options_columns, 
                                            index=0)
            
                if evaluate_col != 'None':

                    table_evaluator = TableEvaluator(original, comparison)
                    evaluate = table_evaluator.evaluate(target_col = evaluate_col)

                else:

                    st.sidebar.warning('Please select a categorical column to analyse.')

        else:

            st.sidebar.warning("Please upload a reference/original dataset.")

def analyse_file():

    """
    The portion of structured data app dedicated to 1 file analysis
    """

    data = st.sidebar.file_uploader("Upload dataset", 
                            type="csv") 

    if data:
        st.subheader('A preview of input data is below, please wait for data to be analyzed :bar_chart:')
        data = pd.read_csv(data)
        st.write(data.head(5))
        
        # generate a report and save it 
        st.session_state['pr'] = data.profile_report()
        pr = st.session_state['pr']
        st_profile_report(pr)
        pr.to_file("pandas_prof.html")
        
        create_pdf_html("pandas_prof.html",
                        "Step 3",
                        "pandas_profiling_dqw.pdf")

def detect_unfairness():
    """  
    Use the fat-forensics package to assess fairness of data
    """
def select_features():
    """
    Automated preprocessing of the structured dataset w/ pycaret
    """