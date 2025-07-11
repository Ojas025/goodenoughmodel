import streamlit as st
from shared.config import dataset_list
from utils.datasets import prettify_dataset_name,read_data_from_file_name,read_data_from_uploaded_file  

def dataset_page():
    st.subheader("Begin with a Dataset...")

    dataset = st.selectbox(
        "Select a dataset",
        dataset_list,
        index=None,
        placeholder = st.session_state.get("name")
    )

    st.markdown("<h4 style='text-align: center;'>OR</h4>",unsafe_allow_html=True)

    uploaded_file = st.file_uploader('Upload Custom Dataset')
    

    if uploaded_file is not None:
        if st.session_state.get("name") != uploaded_file.name:
            st.session_state.df = read_data_from_uploaded_file(uploaded_file)
            st.session_state.name = uploaded_file.name 
            st.session_state.target = None  
            st.rerun()          
        
    elif dataset:
        if st.session_state.get("name") != dataset: 
            st.session_state.df = read_data_from_file_name(dataset)
            st.session_state.name = dataset 
            st.session_state.target = None            
            st.rerun()  

    df = st.session_state.get("df")        
    name = st.session_state.get("name")        

    if df is not None:
        
        processed_name = prettify_dataset_name(name)
        
        st.header(processed_name)
        st.dataframe(df.head())
        
        st.selectbox(
            "Select Target Feature",
            options=df.columns,
            key='target'
        )