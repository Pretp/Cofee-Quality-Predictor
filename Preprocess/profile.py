import pandas as pd

df = pd.read_csv("/Users/prabhpreet16/Documents/IML-Project/merged_data_cleaned.csv")

def profiledata(df) :
    
    print("basic info: ", df.info())  
    
    print("basic stats: ", df.describe())
    
    print("missing values: ", df.isnull().sum())
    
    print("data types: ", df.dtypes)
    
    print("unique values in a column: ", df.nunique())
    
    print("first 5 rows: ", df.head())
    
profiledata(df)
    
    
    
    
    
    