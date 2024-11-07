import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def pie_chart_page(df):
    st.title("Pie Chart")

    # Check and display data types
    st.write("Data Types in the Dataset:")
    st.write(df.dtypes)

    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if categorical_columns:
        selected_column = st.sidebar.selectbox("Select a categorical column for Pie Chart:", categorical_columns)

        # Get value counts for the selected column
        pie_data = df[selected_column].value_counts()

        # Plotting the Pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(plt)
    else:
        st.write("No categorical columns available for pie chart. Please ensure your dataset has categorical columns.")
        st.write("Consider converting text columns to 'category' type for better handling.")
