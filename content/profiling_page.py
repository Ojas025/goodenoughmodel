import streamlit as st
from utils.datasets import prettify_dataset_name
import matplotlib.pyplot as plt
import seaborn as sns 

'''
    Dataset Overview:
        - Number of features
        - Number of rows
        - Missing Value Count
        - Duplicate data points
        - Data Types for each feature
    
    Data Cleaning:
        - Identify Outliers (box plots)
        - Handle Missing Values (Impute or Drop)
        - Remove Duplicates
        - Convert Data Types 
    
    Exploratory Data Analysis
        - Stats (mean,median,mode,min,max)
        - Distribution plots (histogram,scatterplot)
        - Correlation Matrix
        - Frequency Distribution (categorical) 
        - Different Visualizations
        
    Target Variable Profiling         
'''


def profiling_page():
    df = st.session_state.get("df")
    name = st.session_state.get("name")
    target = st.session_state.get("target")
    
    if df is not None:
        
        st.header(prettify_dataset_name(name), "Dataset")
        
        st.subheader("Dataset Overview")

        # Shape
        st.write("Number of Features:",df.shape[1] - 1)
        st.write("Number of Data Points:",df.shape[0])

        # Missing Values
        st.write("Missing Values")
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0])
        
        # Duplicate Values
        st.write("Duplicate Values")
        
        duplicate_values = df.duplicated()
        st.dataframe(df[duplicate_values])
        
        st.write(f"Number of Duplicates: {duplicate_values.sum()}")
        
        # Data Types
        st.write("Data Types")
        st.write(df.dtypes)
        
        # Statistical Summary
        st.write("Statistical Summary")
        st.write(df.describe())
        
        # Feature Distribution
        numeric_features = df.select_dtypes(include=['int64','float64']).columns
        
        if len(numeric_features) > 0:
            st.write("Feature Distribution (Numeric)")
            numeric_feature_to_plot = st.selectbox("Choose a feature",numeric_features)
            fig,ax = plt.subplots()
            sns.histplot(df[numeric_feature_to_plot],ax=ax,kde=True)
            st.pyplot(fig)
        
        # Correlation Matrix
        st.write("Correlation Matrix")
        corr = df.corr()
        fig, ax = plt.subplots() 
        sns.heatmap(corr,ax=ax,cmap='coolwarm',annot=True,annot_kws={ "fontsize": 4 })
        st.pyplot(fig)

        # Frequency Distribution for Categorical Features
        categorical_features = df.select_dtypes(include=['object','category']).columns 
        
        if len(categorical_features) > 0 :
            st.write("Frequency Distribution of Categories")
            feature_to_plot = st.selectbox("Choose a feature",categorical_features)
            st.bar_chart(df[feature_to_plot].value_counts())
        
            
        # Categories for each feature
        if len(categorical_features) > 0:
            st.write("Unique Categories")
        
        for feature in categorical_features:
            st.write(f"Categories in {feature}")
            for category in df[feature].unique():
                st.markdown(f"\t- {category}")
                
        # Outliers Visualization
        if len(numeric_features) > 0:
            st.write("Outlier Visualization")
            outlier_feature_to_plot = st.selectbox("Choose a feature",numeric_features)
            fig,ax = plt.subplots()
            sns.boxplot(x=df[outlier_feature_to_plot],ax=ax)
            ax.set_title(f"Boxplot of {outlier_feature_to_plot}")
            st.pyplot(fig) 
        
        # Target Distribution
        if target:
            st.write("Target Feature Distribution")
            st.bar_chart(df[target].value_counts()) 
        
    else:
        st.write("Please upload or select a dataset.")        