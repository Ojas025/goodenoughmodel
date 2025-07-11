import streamlit as st

def configuration_page():
    
    # Missing Values
    st.session_state.missing_numeric_method = st.selectbox("Missing Value Handling (numeric)", options=['mean','median','drop'],index=0)
    st.session_state.missing_categorical_method = st.selectbox("Missing Value Handling (categorical)", options=['mode','drop'],index=0)
    
    # Add different encoding methods
    st.session_state.scaling_technique = st.radio("Scaling Technique",options=["StandardScaler","MinMaxScaler","None"],index=0,horizontal=True)
    
    drop_corr_flag = st.radio("Drop highly correlated features?", options=["Yes","No"],horizontal=True)
    
    if drop_corr_flag == "Yes":
        st.session_state.correlation_threshold = st.slider("Correlation Threshold",0.0,1.0,0.0,step=0.05)
    
    st.session_state.imbalance_handling_technique = st.radio("Method to handle Imbalance", options=["SMOTE","UnderSampling","None"],horizontal=True,index=2)         
    
    st.session_state.outlier_handling_method = st.radio("Method to handle outliers",options=["IQR","Z-Score","None"],horizontal=True,index=2)
    
    st.session_state.hpt = st.radio("Hyperparameter Tuning", options=["GridSearchCV","RandomisedSearchCV","None"],horizontal=True,index=0)
    
    st.session_state.scoring_metric = st.selectbox("Scoring Metric",options=["accuracy","r2_score","precision","recall","f1","roc_auc_score"],index=0)