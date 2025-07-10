import sklearn
import sklearn.datasets
import pandas as pd

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

load_dataset_from_sklearn()        