import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    """Load and preprocess the diamond dataset."""
    df = pd.read_csv(filepath)
    df = df.drop(labels=['id'], axis=1)
    return df

def remove_outliers(df):
    """Remove outliers from the dataset."""
    df = df.drop(df[df["x"]==0].index)
    df = df.drop(df[df["y"]==0].index)
    df = df.drop(df[df["z"]==0].index)
    
    df = df[(df["depth"]<70.0) & (df["depth"]>54.0)]
    df = df[(df["table"]<73) & (df["table"]>50)]
    df = df[(df["x"]>2)]
    df = df[(df["y"]<9)]
    df = df[(df["z"]<6) & (df["z"]>2)]
    
    return df

def encode_categorical(df):
    """Encode categorical variables using LabelEncoder."""
    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoder = LabelEncoder()
    
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    return df