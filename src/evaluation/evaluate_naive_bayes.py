import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

from src.models.naive_bayes import NaiveBayesClassifier
from src.training.train_naive_bayes import load_and_process_data

def main():
    data_path = os.path.join(project_root, 'data', 'raw', 'adult.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        return

    print("Loading data for final evaluation...")
    X, y = load_and_process_data(data_path)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    X_final_train = np.vstack((X_train, X_val))
    y_final_train = np.concatenate((y_train, y_val))
    
    print("Training final model on Train + Validation sets...")
    best_alpha = 1.0 
    model = NaiveBayesClassifier(alpha=best_alpha)
    model.fit(X_final_train, y_final_train)
    
    print("Predicting on Test Set...")
    y_pred = model.predict(X_test)
    
    print("\n" + "="*40)
    print("FINAL NAIVE BAYES EVALUATION REPORT")
    print("="*40)
    print(f"Final Test Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()