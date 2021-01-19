import streamlit as st
import pandas as pd
import numpy as np

@st.cache
def get_data():
    news = pd.read_csv('abcnews-date-text.csv',nrows=10000)
    return news

try:
    news = get_data()
    plots = st.multiselect(
        "Choose plot", list("Plot1"), ["Plot1"]
    )
    if not plots:
        st.error("Please select at least one plot.")
    else:
        
        st.write("### Length of text")
        
        plot = (
            news['headline_text'].str.len().hist()
        )
        st.altair_chart(plot, use_container_width=True)
except urllib.error.URLError as e:
    st.error(
        """
        **This demo requires internet access.**

        Connection error: %s
    """
        % e.reason
    )