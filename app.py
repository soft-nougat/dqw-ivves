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

    # app design
    app_meta('📊')
    set_bg_hack('dqw_background.png')

    
    # hide warning for st.pyplot() deprecation
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # Main panel setup
    display_app_header(main_txt='Data Quality Wrapper',
                       sub_txt='Clean, describe, visualise and select data for AI models')
    
    st.markdown("""---""")
    
    st.write('Welcome to the DQW! An app for automated preprocessing of data.')
    
    st.write('Please select the data format/app section you want to use below. ',
    'Due to the multifunctionality of this app, we have split it into five.',
    'This is the main app. Have fun!')
    
    # provide options to user to navigate to other dqw apps
    app_section_button('[Image Data Section 🖼️](https://share.streamlit.io/soft-nougat/dqw-ivves_images/main/app.py)',
    '[Tabular Data Section 🏗️](https://share.streamlit.io/soft-nougat/dqw-ivves_structured/main/app.py)',
    '[Audio Data Section 🎶](https://share.streamlit.io/soft-nougat/dqw-ivves_audio/main/app.py)',
    '[Text Data Section 📚](https://share.streamlit.io/soft-nougat/dqw-ivves_text/main/app.py)')
    st.markdown("""---""")

    intro_text = """
    The DQW is an accelerator for the Quality AI Framework.
    Inspired by the CRISP-DM Framework, <b> QAIF </b> is a cohesive, generic framework that 
    can be tailored to a specific AI solution in a given context. 
    The framework is comprised of six gates that follow the process flow of the 
    AI project development cycle (CRISP-DM). The gates can be broken down into project phase, 
    processes, outcomes, governance and people. In each gate, there are specific tasks that need 
    to be completed for the gate to be passed through in order to enter the next gate.
    This ensures that each phase of the AI development cycle is validated thoroughly.
    
    You can access the app <a href = "https://share.streamlit.io/soft-nougat/sogeti_qaif/main/qaif_app.py"> 
    here </a>, to read about the Data Understanding and Preparation phase we 
    use the DQW for.

    """
    intro = st.expander("Click here for more info on quality AI in Sogeti ✨")

    with intro:
        sub_text(intro_text)

    data_bias = """
        Prepare data for model training and ensure the training dataset is not biased.
        This can be achieved by the following methods:
        <li><span style = "color:#F26531">
        <dfn title = 
        "Stratified random sampling is one common method that is used by researchers because it
        enables them to obtain a sample population that best represents the entire population being
        studied, making sure that each subgroup of interest is represented.">
        - Stratified random sampling</span></dfn></li>
        <li><span style = "color:#F26531">
        <dfn title =
        "Oversampling can be used to avoid sampling bias in situations where members of defined
        groups are underrepresented (undercoverage). This is a method of selecting respondents from
        some groups so that they make up a larger share of a sample than they actually do the
        their population. After all data is collected, responses from oversampled groups are weighted to 
        actual share of the population to remove any sampling bias.">
        - Oversampling</span></dfn></li>
        <li><span style = "color:#F26531">
        <dfn title =
        "Sampling weights are intended to compensate for the selection of specific observations 
        with unequal probabilities (oversampling), non-coverage, non-responses, and other types of 
        bias. If a biased data set is not adjusted, population descriptors (e.g., mean, median) 
        will be skewed and fail to correctly represent the population’s proportion to the population.">
        - Adjusting weights</span></dfn></li>
        <li><span style = "color:#F26531">
        <dfn title = "Define a target population and a sampling frame (the list of individuals that 
        the sample will be drawn from). Match the sampling frame to the target population 
        as much as possible to reduce the risk of sampling bias.">
        - Estimate missing data of classes</span></dfn></li>
        <br>In the Structured data app, you can use PyCaret's class imbalance mitigation to 
        solve issues in structured data bias.      
        <br>Text data bias starts with assessing the documents for biased content by a human. 
        Then, you can use do an analysis on the word embeddings. If they carry bias,
        the model may give biased output. We are still working on adding this to the text data
        app. For now, we recommend checking out <a href = 'https://github.com/tdavislab/verb'> 
        VERB </a>, an open-source package for visualisation of embeddings for bias estimation.
    """
    bias = st.expander("Click here for more info on data bias and how to fix it 🔧")

    with bias:
        sub_text(data_bias)

    

except KeyError:
    st.error("Please select a key value from the dropdown to continue.")
    
except ValueError:
    st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
    
except TypeError:
    st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
