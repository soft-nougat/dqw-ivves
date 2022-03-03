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
    
    st.write('Please select the data format/app section you want to use in the main window. ',
    'Due to the multifunctionality of this app, we have split it into five.',
    'This is the main app. Have fun!')
    
    # provide options to user to navigate to other dqw apps
    app_section_button('[Image Data Section 🖼️](https://share.streamlit.io/soft-nougat/dqw-ivves_images/main/app.py)',
    '[Tabular Data Section 🏗️](https://share.streamlit.io/soft-nougat/dqw-ivves_structured/main/app.py)',
    '[Audio Data Section 🎶](https://share.streamlit.io/soft-nougat/dqw-ivves_audio/main/app.py)',
    '[Text Data Section 📚](https://share.streamlit.io/soft-nougat/dqw-ivves_text/main/app.py)')
    st.markdown("""---""")
    

except KeyError:
    st.error("Please select a key value from the dropdown to continue.")
    
except ValueError:
    st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
    
except TypeError:
    st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
