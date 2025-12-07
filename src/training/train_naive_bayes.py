import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

from src.models.naive_bayes import NaiveBayesClassifier

def load_and_process_data(filepath):
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
               'marital-status', 'occupation', 'relationship', 'race', 'sex', 
               'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    
    df = pd.read_csv(filepath, names=columns, na_values=' ?', skipinitialspace=True)
    
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                        'relationship', 'race', 'sex', 'native-country']
    target_col = 'income'
    
    df_clean = df[categorical_cols + [target_col]].copy()
    
    df_clean.fillna('Missing', inplace=True)
    
    label_encoders = {}
    for col in df_clean.columns:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le
        
    X = df_clean.drop(columns=['income']).values
    y = df_clean['income'].values
    
    return X, y

def main():
    data_path = os.path.join(project_root, 'data', 'raw', 'adult.csv')
    
    print(f"Looking for data at: {data_path}")
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        return

    print("Loading and processing data...")
    X, y = load_and_process_data(data_path)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Training data shape: {X_train.shape}")
    
    print("Training Naive Bayes Model...")
    model = NaiveBayesClassifier(alpha=1.0) 
    model.fit(X_train, y_train)
    
    print("Evaluating on Validation Set...")
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    
    print(f"Training Complete.")
    print(f"Validation Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()