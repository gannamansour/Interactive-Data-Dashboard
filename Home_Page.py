import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu  
from contextlib import contextmanager, redirect_stdout
from io import StringIO

st.write("Data Dashboard For California Houses")

    # Output in Web Streamlit :
@contextmanager
def st_capture(output_func):
                with StringIO() as stdout, redirect_stdout(stdout):
                    old_write = stdout.write

                    def new_write(string):
                        ret = old_write(string)
                        output_func(stdout.getvalue())
                        return ret
                    
                    stdout.write = new_write
                    yield


with st.sidebar:
    selected = option_menu(
        menu_title="User's Input For Data Preparation",
        options=[
            "California Houses",
            "Telescope Data",
        ],
    )

if selected == "Telescope Data":
    st.title("Dashboard For Telescope Data")
    data = pd.read_csv(r"D:\Knn-LinearRegression-main\Telescope_data.csv")
    data = pd.DataFrame(data)
    st.write("You selected All Data")
    st.dataframe(data)

if selected == "California Houses":
    st.title("Dashboard For California Houses")
    data = pd.read_csv(r"D:\Knn-LinearRegression-main\California_Houses.csv")
    data = pd.DataFrame(data)
    st.write("You selected All Data")
    st.dataframe(data)       
