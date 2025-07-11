import streamlit as st
import pandas as pd

from utils.datasets import read_data_from_file_name,prettify_dataset_name,read_data_from_uploaded_file
from content.dataset_page import dataset_page
from content.profiling_page import profiling_page
from content.download_model_page import download_model_page
from content.model_training_page import model_training_page
from content.preprocessing_page import preprocessing_page
from content.configuration_page import configuration_page

st.set_page_config(
    page_title='Good Enough Model: Automated Machine Learning',
    layout='wide',
    initial_sidebar_state='expanded',
    page_icon='assets/image-v2.png'
)

df = st.session_state.get('df')
name = st.session_state.get('name') 
target = st.session_state.get('target')
page = "Dataset"

page_list = {
    "Dataset": dataset_page,
    "Profiling": profiling_page,
    "Configuration": configuration_page,
    "Preprocessing": preprocessing_page,
    "Model Training": model_training_page,
    "Download Model": download_model_page,
}

st.sidebar.empty()

with st.sidebar:
    st.image('assets/image-v2.png',caption='Good Enough Model', width=160)
    
    page = st.radio(
        "Navigation",
        options=["Dataset","Profiling","Configuration","Model Training","Download Model"]
    )

st.title("Automated ML Model Training")
st.markdown("---")

page_list[page]()

st.markdown("---")

    