import streamlit as st
from sklearn.preprocessing import StandardScaler,MinMaxScaler 
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import zscore

def preprocessing_page():
    df = st.session_state.get("df")
    target = st.session_state.get("target")
    name = st.session_state.get("name")
    missing_numeric_method = st.session_state.get("missing_numeric_method","mean") 
    missing_categorical_method = st.session_state.get("missing_categorical_method","mode") 
    correlation_threshold = st.session_state.get("correlation_threshold")
    imbalance_handling_technique = st.session_state.get("imbalance_handling_technique")
    scaling_technique = st.session_state.get("scaling_technique") 
    outlier_handling_method = st.session_state.get("outlier_handling_method") 
    
    numeric_features = df.select_dtypes(include=['int64','float64']).columns
    categorical_features = df.select_dtypes(include=['object','category']).columns
    
    # Handling Missing Values
    if df is not None:
        for feature in numeric_features[df[numeric_features].isnull().any()]:
            df = handle_missing_values(df,feature,'numeric',missing_numeric_method)
        
        for feature in categorical_features[df[categorical_features].isnull().any()]:
            df = handle_missing_values(df,feature,'categorical',missing_categorical_method) 
                     
        st.success("Missing values handled successfully")                     
    else:
        st.warning("No dataset loaded.") 
        
    # Handling Outliers
    df = remove_outliers(df,numeric_features,method=outlier_handling_method)

    numeric_features = df.select_dtypes(include=['int64','float64']).columns
    categorical_features = df.select_dtypes(include=['object','category']).columns
    
    if outlier_handling_method != "None":
        st.info("Handled Outliers in the dataset")  
     

    # Feature Encoding 
    if df is not None:
        df = one_hot_encode_features(df,categorical_features,exclude=[target])
        st.success("Performed One-Hot Encoding")

    # Standardization 
    if df is not None:
        st.session_state.df = standardize(df,scaling_technique,exclude=[target])
        st.success("Standardization applied")

    # Correlation Feature Removal
    if df is not None:
        if correlation_threshold > 0:
            df,collinear_features = drop_highly_correlated_features(df,correlation_threshold)
            
            st.info("Dropped these highly correlated features:",collinear_features) 
            
    # Handling Imbalanced Dataset
    df = handle_imbalanced_data(df,target,imbalance_handling_technique)
    st.info(f"Performed {imbalance_handling_technique} on the dataset for handling imbalance")
    
    st.session_state.df = df
    st.rerun()                              

def handle_missing_values(df,feature_name,dtype,method):
    if dtype == 'numeric':
        if method == 'mean':
            df[feature_name] = df[feature_name].fillna(df[feature_name].mean()) 
        elif method == 'median':
            df[feature_name] = df[feature_name].fillna(df[feature_name].median())
        else:
            df.dropna(subset=[feature_name],inplace=True)                        
    else:
        if method == 'mode':
            df[feature_name] = df[feature_name].fillna(df[feature_name].mode()[0])
        else:
            df.dropna(subset=[feature_name],inplace=True)    
    
    return df            

def standardize(df,method='StandardScaler',exclude=[]):
    if method == 'StandardScaler':
        scaler = StandardScaler()
    elif method == 'MinMaxScaler':
        scaler = MinMaxScaler()
            
    numeric_features = [feature for feature in df.select_dtypes(include=['int64','float64']).columns if feature not in exclude]
    
    if not numeric_features:
        return df
    
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    return df
    
def drop_highly_correlated_features(df,correlation_threshold,target = None):
    corr_matrix = df.select_dtypes(include=['int64','float64']).corr().abs()
    
    if target and target in corr_matrix:
        corr_matrix.drop(columns=target,inplace=True)
    
    collinear_features = set()        
    
    # Store the columns with correlation greater than the given threshold  
    for i in range(corr_matrix.shape[0]):
        for j in range(i+1,corr_matrix.shape[1]):
            if corr_matrix.iloc[i,j] > correlation_threshold:
                collinear_features.add(corr_matrix.columns[j])
    
    # Drop all those columns 
    df = df.drop(columns=collinear_features,axis=1)
    
    return df,list(collinear_features)                              
                         
def one_hot_encode_features(df,features,exclude=[]):
    # Features which are to be one hot encoded, excluding the list provided
    features_to_encode = [feature for feature in features if feature not in exclude] 
    
    # Encode these features 
    df_encoded = pd.get_dummies(df[features_to_encode],drop_first=True)
    
    # Features which were not encoded, includes numeric and target features 
    df_non_encoded = df.drop(columns=features_to_encode)
    
    # Combined final dataset with encoded as well as non-encoded features
    df_final = pd.concat([df_encoded,df_non_encoded],axis=1)
    
    return df_final                         

def handle_imbalanced_data(df,target,method='SMOTE'):
    if method == "None":
        return df

    X = df.drop(columns=[target])
    y = df[target] 

    if method == 'SMOTE':
        smote = SMOTE(random_state=42)
        X_res,y_res = smote.fit_resample(X,y)
    else:  
        undersampler = RandomUnderSampler(random_state=42)
        X_res,y_res = undersampler.fit_resample(X,y)
         
    df_resampled = pd.concat([pd.DataFrame(X_res,columns=X.columns),pd.Series(y_res,name=target)],axis=1)
    return df_resampled

def remove_outliers(df,numeric_features,method='IQR'):
    if method == 'None':
        return df
    elif method == 'IQR':
        # Calculate Q1, 25 percentile
        Q1 = df[numeric_features].quantile(0.25)
        
        # Calculate Q3, 75 percentile
        Q3 = df[numeric_features].quantile(0.75)
        
        # Calulate the Inter-quantile Range
        IQR = Q3 - Q1
        
        # Calculate the upper and lower limits
        upper_limit = Q3 + 1.5 * IQR
        lower_limit = Q1 - 1.5 * IQR
        
        # Drop the data points with values lower than lower_limit and higher than upper_limit 
        outlier_condition = ((df[numeric_features] < lower_limit) | (df[numeric_features] > upper_limit))
        df = df[~outlier_condition.any(axis=1)]  
    else:
        # Compute the z-score for the numeric features
        z_scores = pd.DataFrame(zscore(df[numeric_features]),columns=numeric_features,index=df.index)
        df = df[(z_scores.abs() <= 3).all(axis=1)]
    
    return df        