# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 15:25:48 2024

@author: Fan Yang, Dejing Chen
"""

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score
import json


def load_and_prepare_data(csv_path):
    """Load dataset and display initial information."""
    df = pd.read_csv(csv_path)
    print(df.head())
    print(df.columns.values)
    return df

def feature_selection(df):
    """Select features and target from the dataframe."""
    feature_cols = ['thickness', 'size', 'shape', 'Marg', 'Epith', 'b1', 'nucleoli', 'Mitoses']
    X = df[feature_cols]
    y = df['class']
    return X, y

def perform_grid_search(X_train, y_train):
    """Define and run GridSearchCV to find the best model parameters."""
    param_grid = {
        'C': [0.1, 1.0, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001, 10],
        'kernel': ['linear', 'rbf']
    }
    grid = GridSearchCV(estimator=SVC(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)

    return grid.best_estimator_, grid.best_params_

def save_model(model, model_save_path):
    """Save the trained model to a file."""
    joblib.dump(model, model_save_path)
    print("Model saved to", model_save_path)

if __name__ == "__main__":
    csv_path = r"C:\Users\maily\Desktop\GenAi_BPC\breast_cancer.csv"
    model_save_path = r"C:\Users\maily\Desktop\GenAi_BPC\breast_cancer_model.pkl" 
    
    df = load_and_prepare_data(csv_path)
    X, y = feature_selection(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    best_model,best_params  = perform_grid_search(X_train, y_train)
    save_model(best_model, model_save_path)

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    # Save model parameters and metrics
    evaluation_metrics = {
        'best_params': best_params,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }

    with open('model_metrics.json', 'w') as f:
        json.dump(evaluation_metrics, f)
    