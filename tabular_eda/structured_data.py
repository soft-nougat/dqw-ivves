# this script contans structured data
# analysis code

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from tabular_eda.te import *
from helper_functions import display_app_header, generate_zip_structured
import pdfkit as pdf
from fpdf import FPDF
import io

# remove before commiting
path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
config = pdf.configuration(wkhtmltopdf=path_wkhtmltopdf)

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

def create_pdf_html(html, header_text, file_name):

    """
    Funstion to create a pdf from an html file
    """
    # create a pdf file
    pdf_report = pdf.from_file(html, configuration = config)

    # option to download in app
    display_app_header(main_txt = header_text,
                        sub_txt= "Download report",
                        is_sidebar=True)

    st.sidebar.download_button(
            "⬇️",
        data=pdf_report,
        file_name=file_name
    )

def create_pdf(original, comparison):

    # instantiate fpdf class, set orientation and format
    # unit in cm because of PowerPoint
    pdf = FPDF(orientation = 'P', unit = 'cm', format = 'A4')

    # set font and color
    pdf.set_font('Times', 'B', 10)
    # get rgb
    pdf.set_text_color(247, 67, 105)
    pdf.set_margins(-1, -1, -1)

    # add page
    pdf.add_page()
    pdf.image("pdf_files/2_file_structured_bg_page1.png", x = 0, y = 0, w = 21, h = 29.7)
    # first page of this report contains 2 plots, add text and plots at positions
    #pdf.write(txt = "This report contains comparison based on the table-evaluator package. The tables analysed can be found in the same zip as this report.",
    #h = 5.93)
    #pdf.text(txt = "Below are the mean and standard deviations of numerical columns in the reference and comparison dataset.",
    #x = 6.03, y = 1.51)
    pdf.image("pdf_files/mean_std.png", x = 6.03, y = 2.59, w = 14.61, h = 7.31)
    #pdf.text(txt = "Below is the comparison of feature distributions for the reference and comparison dataset.",
    #x = 6.03, y = 10.07)
    pdf.image("pdf_files/distributions.png", x = 5.51, y = 12.16, w = 15.17, h = 17.06)

    pdf.add_page()
    pdf.image("pdf_files/2_file_structured_bg_page2.png", x = 0, y = 0, w = 21, h = 29.7)
    #pdf.text(txt = "Below are plotted correlations between categorical columns of the reference and comparison dataset.",
    #x = 6.03, y = 0.76)
    #pdf.text(txt = "Reference dataset correlation heatmap",
    #x = 7.77, y = 1.88)
    pdf.image("pdf_files/corr_ref.png", x = 5.43, y = 2.65, w = 11.74, h = 8.81)
    #pdf.text(txt = "Comparison dataset correlation heatmap",
    #x = 7.77, y = 11.2)
    pdf.image("pdf_files/corr_comp.png", x = 5.83, y = 11.84, w = 11.35, h = 8.51)
    #pdf.text(txt = "Differences",
    #x = 7.77, y = 19.93)
    pdf.image("pdf_files/corr_diff.png", x = 6.42, y = 20.7, w = 11.35, h = 8.51)

    pdf.add_page()
    pdf.image("pdf_files/2_file_structured_bg_page3.png", x = 0, y = 0, w = 21, h = 29.7)
    #pdf.text(txt = "Below is the PCA result for reference and comparison dataset.",
    #x = 6.03, y = 0.76)
    pdf.image("pdf_files/pca.png", x = 4.46, y = 2.46, w = 17.71, h = 8.86)

    pdf.output("pdf_files/table-evaluator_comparison_dqw.pdf")