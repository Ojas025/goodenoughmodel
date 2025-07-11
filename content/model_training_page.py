import streamlit as st
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,plot_tree
from sklearn.metrics import r2_score,accuracy_score,mean_absolute_error,mean_squared_error,confusion_matrix,classification_report,roc_auc_score,roc_curve
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from xgboost import XGBClassifier,XGBRegressor
import numpy as np
import pandas as pd

from shared.parameters import *

classification_models = {
    'LogisticRegression': LogisticRegression,
    'DecisionTreesClassifier': DecisionTreeClassifier,
    'XGBClassifier': XGBClassifier,
    'RandomForestClassifier': RandomForestClassifier
}

regression_models = {
    'LinearRegression': LinearRegression,
    'DecisionTreesRegressor': DecisionTreeRegressor,
    'XGBRegression': XGBRegressor,
    'RandomForestRegression': RandomForestRegressor
}

def model_training_page():
    df = st.session_state.get("df")
    target = st.session_state.get("target")
    name = st.session_state.get("name")
    hpt = st.session_state.get("hpt")
    scoring_metric = st.session_state.get("scoring_metric")
     
    if df is None:
        st.write("Select or upload a dataset.")
        return 
    
    
    X = df.drop(columns=[target])
    y = df[target]
    
    problem_type = infer_problem_type(y)

    # Split the dataset
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    if problem_type == 'regression':
        train_model(X_train,X_test,y_train,y_test,regression_models,hpt,problem_type,scoring_metric)
    else:
        train_model(X_train,X_test,y_train,y_test,classification_models,hpt,problem_type,scoring_metric) 

    # Initialize the model
    
    # Hyper Parameter Tuning
    
    # Train the model
    
    # Test the model
    
    # Compute efficiency of the model  

def infer_problem_type(y):
    if pd.api.types.is_object_dtype(y):
        return "classification"
    
    unique_values = len(y.unique())
    
    if unique_values == 2:
        return "classification"
    
    if pd.api.types.is_numeric_dtype(y) and unique_values <= 10:
        return "classification"
    
    return "regression"

def train_model(X_train,X_test,y_train,y_test,models,hpt,problem_type,scoring_metric):
    '''
        Instantiate model
        Instiantiate grid_search | randomised_search
        hyper parameter train
        train the model with best parameters
        predict on test data
        display model metrics 
    '''
    
    trained_models = []
    predictions = []
    names = []
    
    for name,model_class in models.items():
        model = model_class()
        
        # Returns grid_search or randomised_search
        # Add Regularization for LinearRegression Later
        hpt_instance = get_hpt(name,model,hpt,scoring_metric) if name != "LinearRegression" else model
        
        # Train the model on the best params 
        hpt_instance.fit(X_train,y_train)
        
        y_pred = hpt_instance.predict(X_test)
        
        names.append(name) 
        trained_models.append(hpt_instance)
        predictions.append(y_pred)
    
    # At this stage, i have all the trained models and their predictions  
    scores = evaluate_model_performance(trained_models,predictions,problem_type)
    
    best_index = np.argmax(scores)
    best_model = trained_models[best_index]
    
    if names[best_index] != 'LinearRegression':
        best_params = best_model.best_params_     
    
    return best_model,best_params   


def get_hpt(name,model,hpt,scoring_metric):
    param_grid = select_param_grid(name,hpt) 

    if hpt == 'None':
        return model
    elif hpt == 'GridSearchCV':
        hpt_instance = GridSearchCV(estimator=model,param_grid=param_grid,refit=True,cv=5,scoring=scoring_metric,n_jobs=-1)
    else:
        hpt_instance = RandomizedSearchCV(estimator=model,param_distributions=param_grid,cv=5,n_jobs=-1,refit=True,scoring=scoring_metric)

    return hpt_instance 

def evaluate_model_performance(models,predictions,problem_type):
    pass

def select_param_grid(name,hpt):
    pass