import streamlit as st

def create_menu(menu_options:list=["Preparation","Selection","Description"]):

    sideb = st.sidebar
    sideb.write("Choose one of the options and get to know your data even better!")
    # menu_options = ["Grid view","Preparation","Selection","Description"]
    menu_buttons = []
    for option in menu_options:
        menu_buttons.append(sideb.button(option))

    for button, option in zip(menu_buttons, menu_options):
        if button == True:
            return option

    # userid=st.sidebar.radio("Choose your user ID",("Grid view","Preparation","Selection","Description"))
