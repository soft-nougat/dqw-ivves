"""
A script containing the logic behind the structured data report generation
"""
import streamlit as st
import pandas as pd
from helper_functions import display_app_header
import pdfkit as pdf
from fpdf import FPDF
import io 

def create_pdf(original, comparison):

    """
    Use fpdf to create a static report, we use files from pdf_files folder
    Pass original and comparison
    """
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
    pdf.image("pdf_files/synthetic_data/2_file_structured_bg_page1.png", x = 0, y = 0, w = 21, h = 29.7)
    # first page of this report contains 2 plots, add text and plots at positions
    #pdf.write(txt = "This report contains comparison based on the table-evaluator package. The tables analysed can be found in the same zip as this report.",
    #h = 5.93)
    #pdf.text(txt = "Below are the mean and standard deviations of numerical columns in the reference and comparison dataset.",
    #x = 6.03, y = 1.51)
    pdf.image("pdf_files/synthetic_data/mean_std.png", x = 6.03, y = 2.59, w = 14.61, h = 7.31)
    #pdf.text(txt = "Below is the comparison of feature distributions for the reference and comparison dataset.",
    #x = 6.03, y = 10.07)
    pdf.image("pdf_files/synthetic_data/distributions.png", x = 5.51, y = 12.16, w = 15.17, h = 17.06)

    pdf.add_page()
    pdf.image("pdf_files/synthetic_data/2_file_structured_bg_page2.png", x = 0, y = 0, w = 21, h = 29.7)
    #pdf.text(txt = "Below are plotted correlations between categorical columns of the reference and comparison dataset.",
    #x = 6.03, y = 0.76)
    #pdf.text(txt = "Reference dataset correlation heatmap",
    #x = 7.77, y = 1.88)
    pdf.image("pdf_files/synthetic_data/corr_ref.png", x = 5.43, y = 2.65, w = 11.74, h = 8.81)
    #pdf.text(txt = "Comparison dataset correlation heatmap",
    #x = 7.77, y = 11.2)
    pdf.image("pdf_files/synthetic_data/corr_comp.png", x = 5.83, y = 11.84, w = 11.35, h = 8.51)
    #pdf.text(txt = "Differences",
    #x = 7.77, y = 19.93)
    pdf.image("pdf_files/synthetic_data/corr_diff.png", x = 6.42, y = 20.7, w = 11.35, h = 8.51)

    pdf.add_page()
    pdf.image("pdf_files/synthetic_data/2_file_structured_bg_page3.png", x = 0, y = 0, w = 21, h = 29.7)
    #pdf.text(txt = "Below is the PCA result for reference and comparison dataset.",
    #x = 6.03, y = 0.76)
    pdf.image("pdf_files/synthetic_data/pca.png", x = 4.46, y = 2.46, w = 17.71, h = 8.86)

    pdf.output("pdf_files/synthetic_data/table-evaluator_comparison_dqw.pdf")

def create_pdf_html(html, header_text, file_name):

    """
    Function to create a pdf from an html file

    Args:
    html - Pass the pandas profiling/sweetviz report and convert to pdf
    header_text - Text to display on the sidebar
    file_name
    """
    # create a pdf file
    pdf_report = pdf.from_file(html)

    # option to download in app
    display_app_header(main_txt = header_text,
                        sub_txt= "Download report",
                        is_sidebar=True)

    st.sidebar.download_button(
            "⬇️",
        data=pdf_report,
        file_name=file_name
    )