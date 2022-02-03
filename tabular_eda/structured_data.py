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
import sweetviz as sv
import pycaret as pyc

def structured_data_app():

    # ----------------------------------------------
    # session state
    if 'pr' not in st.session_state:
        st.session_state.pr = None

    if 'sw' not in st.session_state:
        st.session_state.sw = None

    if 'data' not in st.session_state:
        st.session_state.data = None
    
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
                                                "Compare 2 files",
                                                "Synthetic data comparison"))

    display_app_header(main_txt = "Step 2",
                    sub_txt= "Upload data",
                    is_sidebar=True)

    
    if selected_structure == "Analyse 1 file":

        st.session_state.data = upload_file()

        if st.session_state.data is not None:

            display_app_header(main_txt = "Step 3",
                            sub_txt= "Choose next step",
                            is_sidebar=True)

            step_3 = st.sidebar.selectbox("",
            ("None", "EDA", "Preprocess and compare"))

            if step_3 == "EDA":

                st.session_state.pr = analyse_file(st.session_state.data)

            elif step_3 == "Preprocess and compare":

                st.session_state.data = preprocess(st.session_state.data)

            else:

                st.warning("Please select next step in sidebar.")
           
    if selected_structure == "Compare 2 files":
        
        st.session_state.sw = sweetviz_comparison(None, None, 0)
    
    if selected_structure == "Synthetic data comparison":
        
        table_evaluator_comparison()

def upload_file():

    data = st.sidebar.file_uploader("Upload dataset", 
                            type="csv") 

    if data:

        st.subheader('A preview of input data is below, please wait for data to be analyzed :bar_chart:')
        data = pd.read_csv(data)
        st.write(data.head(5))

        return(data)

    else:
        st.sidebar.warning("Please upload a dataset!")

        return(None)
        



            
def upload_2_files():

    """
    High level app logic when comparing 2 files
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

            return(original, comparison, 1)

    else:
        st.sidebar.warning("Please upload a reference/original dataset.")

        return(None, None, 0)

def sweetviz_comparison(original, comparison, indicator, upload = True):

    """
    Function to compare test and train data with sweetviz
    """
    if upload == True:
        # call high level function and get files
        original, comparison, indicator = upload_2_files()

    # use indicator to stop the app from running
    if indicator == 1: 

        sw = sv.compare([original, "Original"], [comparison, "Comparison"])

        sw.show_html(open_browser=False, layout='vertical', scale=1.0)

        display = open("SWEETVIZ_REPORT.html", 'r', encoding='utf-8')

        source_code = display.read()

        components.html(source_code, height=1200, scrolling=True)

        create_pdf_html("SWEETVIZ_REPORT.html",
                        "Step 3",
                        "sweetviz_dqw.pdf")

        return(sw)


def table_evaluator_comparison():

    """
    The portion of structured data app dedicated to file comparison with table-evaluator
    We have 2 options, plot differences or choose categorical column to analyse
    """

    # call high level function and get files
    original, comparison, indicator = upload_2_files()

    if indicator == 1: 
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
                    file_name="te_compare_files_dqw.zip",
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

        
def analyse_file(data):

    """
    The portion of structured data app dedicated to 1 file analysis
    """
    
    # generate a report and save it 
    st.session_state['pr'] = data.profile_report()
    pr = st.session_state['pr']
    st_profile_report(pr)
    pr.to_file("pandas_prof.html")
    
    create_pdf_html("pandas_prof.html",
                    "Step 4",
                    "pandas_profiling_dqw.pdf")

    return(pr)


def preprocess(data):
    """
    Automated preprocessing of the structured dataset w/ pycaret
    """
    # class column is the label - ask the user to select - not necessary for unsupervised
    model = st.selectbox('Select the type of model you are preparing data for:',
    ('Unsupervised', 'Supervised'))

    # unsupervised
    if model == 'Unsupervised':

        from pycaret.clustering import setup, get_config
        clf_unsup = setup(data = data, silent = True)

    # superivised
    else:

        from pycaret.classification import setup,  get_config
        dataset_columns = data.columns
        options_columns = dataset_columns.insert(0, 'None')
        
        label_col = st.selectbox('Select the label column:', 
                                    options_columns, 
                                    index=0)

    
        if label_col != 'None':

            clf_sup = setup(data = data, target = label_col, silent = True)


    dataset_x = get_config('X')

    st.write(dataset_x.head(5))

    st.header("Compare files")

    compare_type = st.selectbox('Select which files to compare:',
    ('Original & preprocessed', 'Train & test'))

    if compare_type == 'Original & preprocessed':

        sweetviz_comparison(data, dataset_x, 1, upload = False)
    
    else:

        sweetviz_comparison(get_config('X_train'), get_config('X_test'), 1, upload = False)

        


def detect_unfairness():
    """  
    Use the fat-forensics package to assess fairness of data
    """
    