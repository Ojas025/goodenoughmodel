import sklearn
import sklearn.datasets
import pandas as pd
from shared.paths import parent_dir
import os

def load_dataset_from_sklearn():
    datasets = {
        'iris': sklearn.datasets.load_iris,
        'breast_cancer': sklearn.datasets.load_breast_cancer,
        'wine': sklearn.datasets.load_wine,
        'diabetes': sklearn.datasets.load_diabetes,
    }
    
    for dataset_name,loader in datasets.items():
        
        dataset = loader()
        X = pd.DataFrame(dataset.data,columns=dataset.feature_names)
        
        y = pd.DataFrame(dataset.target,columns=['target'])
        
        df = pd.concat([X,y],axis=1)
        
        df.to_csv(f'../datasets/raw/{dataset_name}.csv',index=False)
        print(f"Saved {dataset_name}!")

def read_data_from_file_name(file_name):
    file_path = os.path.join(parent_dir, "datasets","raw",file_name)
    
    if file_name is None:
        return None 
    
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    else:
        return None       
    
def read_data_from_uploaded_file(file):
    if file is None:
        return None 
    
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
        return pd.read_excel(file)
    else:
        return None       
    

def prettify_dataset_name(dataset):
    '''
    Input => some_dataset.csv
    Output => Some Dataset 
    '''                 
    
    name = dataset.split('.')[0]
    
    name = ' '.join([ i.title() for i in name.split('_')])
    
    return name