"""

A file with the high level image eda app functions

"""
import streamlit as st
from helper_functions import *
from image_eda.preparation import show_grid, show_sizes, show_histograms, show_channels, augmentations, export_images
    
   
def image_data_app():

    st.write("Welcome to the DQW for unsupervised image data. ",
            "Understanding your data is an important step ",
            "in AI model development. This app ",
            "offers visualisation of descriptive statistics of a ",
            "collection of images using a multitude of packages.")

    display_app_header(main_txt = "Step 1",
                       sub_txt= "Upload data",
                       is_sidebar=True)

    images = load_images()

    if images:
    
        display_app_header(main_txt = "Step 2",
                        sub_txt= "Choose what you want to see/do",
                        is_sidebar=True)

        selected_structure = st.sidebar.radio(
                            "", 
                            ("Image grid", 
                            "Image sizes",
                            "Color histograms",
                            "Color channels",
                            "Augmentations"))

        if selected_structure == "Image grid":
            show_grid(images)
        elif selected_structure == "Image sizes":
            show_sizes(images)
        elif selected_structure == "Color histograms":
            show_histograms(images)
        elif selected_structure == "Color channels":
            show_channels(images)
        elif selected_structure == "Augmentations":
            augmentations(images)
        
            display_app_header(main_txt = "Step 3",
                            sub_txt= "Export augmented data",
                            is_sidebar=True)

            is_export = st.sidebar.button("Export ⬇️")
            if is_export:
                export_images(images)

    

    # prompt the user with an option to select which data they want to 
    # analyse - save code, right now we only have 1 option
    #selected_structure = st.selectbox("Choose data structure to analyse", 
                                       #("Unsupervised image data", 
                                       #"Supervised image data, Classification",
                                       #"Supervised image data, Segmentation",
                                       #"Supervised image data, Regression"))
 
    #elif selected_structure == "Supervised image data, Classification":
        #st.error("This feature has not been implemented yet!")
    #elif selected_structure == "Supervised image data, Segmentation":
        #st.error("This feature has not been implemented yet!")
    #elif selected_structure == "Supervised image data, Regression":
        #st.error("This feature has not been implemented yet!")