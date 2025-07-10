import streamlit as st
from shared.config import dataset_list
from utils.datasets import prettify_dataset_name,read_data_from_file_name,read_data_from_uploaded_file  

def dataset_page():
    st.subheader("Begin with a Dataset...")

    dataset = st.selectbox(
        "Select a dataset",
        dataset_list,
        index=None
    )

    st.markdown("<h4 style='text-align: center;'>OR</h4>",unsafe_allow_html=True)

    uploaded_file = st.file_uploader('Upload Custom Dataset')

    if uploaded_file is not None:
        df = read_data_from_uploaded_file(uploaded_file)
        name = uploaded_file.name

        if "df" not in st.session_state or st.session_state.df is None:
            st.session_state.df = df
             
        if "name" not in st.session_state or st.session_state.name is None:
            st.session_state.name = name 
        
    elif dataset:
        df = read_data_from_file_name(dataset)
        name = dataset

        if "df" not in st.session_state or st.session_state.df is None:
            st.session_state.df = df
             
        if "name" not in st.session_state or st.session_state.name is None:
            st.session_state.name = name 
    else:
        df = None
        name = None         

    if df is not None:
        
        processed_name = prettify_dataset_name(name)
        
        st.header(processed_name)
        st.dataframe(df.head())
        
        target = st.selectbox(
            "Select Target Feature",
            options=df.columns,
            key='target'
        )
    
        if 'target' not in st.session_state or st.session_state.target is None:
            st.session_state.target = target