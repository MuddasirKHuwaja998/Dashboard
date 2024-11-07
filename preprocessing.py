import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def preprocessing_page(df):
    st.title("Preprocessing")

    # Check if the dataframe is empty
    if df.empty:
        st.write("The dataset is empty.")
        return df

    # Handling missing values
    st.write("Handling missing values...")

    # Impute missing numeric values with the median
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) > 0:
        imputer = SimpleImputer(strategy='median')
        df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    else:
        st.write("No numeric columns to impute.")

    # Impute missing categorical values with the most frequent value
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_columns) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df[categorical_columns] = imputer_cat.fit_transform(df[categorical_columns])
    else:
        st.write("No categorical columns to impute.")

    # Drop any remaining missing values (optional, in case of non-numeric/categorical columns)
    df = df.dropna()

    st.write("Encoding categorical variables...")

    # Encode categorical variables using LabelEncoder
    label_encoders = {}
    for column in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    st.write("Normalizing numeric features...")

    # Normalize numeric features using StandardScaler
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) > 0:
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    else:
        st.write("No numeric columns to normalize.")

    st.write("Data after preprocessing:")
    st.write(df.head())
    
    return df
