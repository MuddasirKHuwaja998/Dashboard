# app.py
import streamlit as st
import pandas as pd

# Set page title and configuration
st.set_page_config(page_title="OtofarmaSPA Dashboard", page_icon=":guardsman:", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
        /* Main page background */
        body {
            background-color: white;
            color: #006400;  /* Dark Green */
            font-family: 'Roboto', sans-serif;
        }

        /* Title styling */
        .stTitle {
            font-size: 34px;
            font-weight: bold;
            color: white !important;
            background-color: #388e3c !important;  /* Green Background */
            font-family: 'Roboto', sans-serif;
            padding: 20px;
        }

        /* Sidebar styling */
        .stSidebar {
            background-color: #388e3c !important; /* Green Sidebar Background */
        }

        .stSidebar .stRadio div label {
            font-size: 18px;
            font-weight: bold;
            color: white !important; /* White Sidebar Text */
        }

        .stSidebar .stTitle {
            font-size: 22px;
            font-weight: bold;
            color: white !important;
        }

        /* File uploader styling */
        .stFileUploader {
            background-color: #000000;
            border: 4px solid #000000;
            border-radius: 10px;
            padding: 15px;
            font-size: 18px;
            text-align: center;
        }

        .stFileUploader > div > label {
            color: #388e3c !important;
            font-weight: 600;
        }

        /* Button styling */
        .stButton {
            background-color: #66bb6a !important;
            color: white !important;
            font-weight: bold;
            padding: 12px 25px;
            border-radius: 8px;
        }

        .stButton:hover {
            background-color: #388e3c !important;
        }

    </style>
""", unsafe_allow_html=True)

# Add logo (Adjust path as needed)
st.image("otofarmaspa_logo.png", width=200)

# Importing pages for navigation
from data_analysis import data_analysis_page
from visualizations import visualizations_page
from pie_chart import pie_chart_page
from preprocessing import preprocessing_page
from predictive_models import predictive_models_page

# Sidebar Navigation
st.sidebar.title("OtofarmaSPA Dashboard")
pages = ["Main Page", "Data Analysis", "Visualizations", "Pie Chart", "Preprocessing", "Predictive Models"]
selected_page = st.sidebar.radio("Select a page:", pages)

# File uploader in sidebar
st.sidebar.title("Upload Your CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Navigation Control
    if selected_page == "Main Page":
        st.title("OtofarmaSPA CSV Data Analyzer Dashboard")
        st.header(f"Data Preview: {uploaded_file.name}")
        st.write(df.head())
    elif selected_page == "Data Analysis":
        data_analysis_page(df)
    elif selected_page == "Visualizations":
        numeric_df = df.select_dtypes(include='number')
        correlation = numeric_df.corr()
        key_variables = correlation.abs().unstack().sort_values(ascending=False).drop_duplicates()
        key_variables = key_variables[key_variables < 1].nlargest(5)
        visualizations_page(df, key_variables)
    elif selected_page == "Pie Chart":
        pie_chart_page(df)
    elif selected_page == "Preprocessing":
        df = preprocessing_page(df)
    elif selected_page == "Predictive Models":
        if df is not None:
            # Debug: Show dataframe columns
            st.write(f"Data Columns: {df.columns}")
            
            # Set target columns as the last three columns
            target_columns = df.columns[-3:]  # Last three columns
            
            # Pass the dataframe and target columns to the predictive models page
            predictive_models_page(df, target_columns)
        else:
            st.warning("Please upload a CSV file first.")
else:
    st.title("OtofarmaSPA CSV Data Analyzer Dashboard")
    st.write("Please upload a CSV file using the sidebar to begin.")
