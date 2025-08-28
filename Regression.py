import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st
from streamlit_option_menu import option_menu  
from contextlib import contextmanager, redirect_stdout
from io import StringIO

def app():
st.title("California Houses")

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

data = pd.read_csv(r"D:\Knn-LinearRegression-main\Pages\California_Houses.csv")
data = pd.DataFrame(data)

with st.sidebar:
    selected = option_menu(
        menu_title="User's Input For Data Preparation",
        options=[
            "Show Data",
            "Linear Regression",
            "Lasso",

        ],
    )

if selected == "Show Data":
        st.title("Dashboard For California Houses")
        st.write("You selected All Data")
        st.dataframe(data)

if selected == "Linear Regression":
        st.title("Linear Regression For California Houses")
        st.title("Choose ")
        with st.sidebar:
        selected_Attri = option_menu(
            menu_title="User's Input For Data Preparation",
            options=[
                    "Median_Income",
                    "Tot_Rooms",
                    "Tot_Bedrooms",
                    "Population",
                    "Households",
                    "Latitude",
                    "Longitude",
                    "Distance_to_coast",
                    "Distance_to_LA",
                    "Distance_to_SanDiego",
                    "Distance_to_SanJose",
                    "Distance_to_SanFrancisco"
                ],
            )
        Button = st.button("Show Linear Regression")
        st.write("Select Number Of Data Headres To Show:")
        Number_Of_Data_Headers = st.slider(
        label='Number Of Data Headers',
        min_value=1,
        max_value=50,
        value=0,
        step=1)


        if Button :
            st.title("Simple Scatter Plot")
            fig, ax = plt.subplots()
            ax.scatter(data['Median_House_Value'].head(Number_Of_Data_Headers),data[selected_Attri].head(Number_Of_Data_Headers))
            ax.set_xlabel("Median House Value")  
            ax.set_ylabel(selected_Attri)  
            st.pyplot(fig)

            x = data['Median_House_Value'].head(Number_Of_Data_Headers)
            y = data[selected_Attri].head(Number_Of_Data_Headers)
            x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=20)
            x_train = np.array(x_train).reshape(-1,1)
            output = st.empty()
            with st_capture(output.code):
                print(x_train)

            lr = LinearRegression ()
            lr.fit(x_train , y_train)
            c = lr.intercept_
            output = st.empty()
            with st_capture(output.code):
                print("Intercepted Part : " , c)

            m = lr.coef_
            with st_capture(output.code):
                print("Slope : " , m)

            st.title("Linear Equations :")
            y_pred_train = m * x_train + c
            with st_capture(output.code):
                print(y_pred_train.flatten())

            y_pred_train1 = lr.predict(x_train)
            with st_capture(output.code):
                print(y_pred_train1)

            st.title("Linear Regression : ")
            fig, ax = plt.subplots()
            ax.scatter(x_train, y_train)
            ax.plot(x_train, y_pred_train1, color='blue')
            ax.set_xlabel("Median House Value") 
            ax.set_ylabel(selected_Attri) 
            st.pyplot(fig)