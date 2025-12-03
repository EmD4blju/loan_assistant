import pandas as pd
from neural_core.models.neural_network import BaseLoanNN, TemperatureScaledNN
import torch
from torch.utils.data import Dataset, DataLoader
from neural_core.models.dataset import DataFrameDataset
from typing import Tuple, Dict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import shap
import matplotlib.pyplot as plt
import numpy as np



def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[Dataset, Dataset]:
    train_dataset = DataFrameDataset(train_df, target_column='loan_status')
    test_dataset = DataFrameDataset(test_df, target_column='loan_status')
    return train_dataset, test_dataset

def evaluate_model(model: BaseLoanNN, test_dataset: Dataset) -> Dict:
    test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model.eval()
    
    accuracy = 0.0
    f1 = 0.0
    roc_auc = 0.0
    precision = 0.0
    recall = 0.0
    
    all_predictions = []
    
    with torch.no_grad():
        for X, y in test_data_loader:
            outputs = model(X)
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()
            
            # Store prediction history
            for i in range(len(X)):
                prob = probabilities[i].item()
                pred_class = int(predicted[i].item())
                confidence = prob if pred_class == 1 else (1 - prob)
                
                all_predictions.append({
                    "sample_number": len(all_predictions),
                    "probability": prob,
                    "predicted_class": pred_class,
                    "confidence": confidence,
                    "true_label": int(y[i].item())
                })
            
            accuracy += accuracy_score(y.numpy(), predicted.numpy())
            f1 += f1_score(y.numpy(), predicted.numpy())
            roc_auc += roc_auc_score(y.numpy(), predicted.numpy())
            precision += precision_score(y.numpy(), predicted.numpy())
            recall += recall_score(y.numpy(), predicted.numpy())
        
    accuracy /= len(test_data_loader)
    f1 /= len(test_data_loader)
    roc_auc /= len(test_data_loader)
    precision /= len(test_data_loader)
    recall /= len(test_data_loader)
    
    results = {
        "metrics": {
            "accuracy": accuracy,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall
        },
        "total_samples": len(all_predictions),
        "predictions": all_predictions
    }
    
    return results
    
def evaluate_temperature_scaled_model(model: TemperatureScaledNN, test_dataset: Dataset) -> Dict:
    test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model.eval()
    
    accuracy = 0.0
    f1 = 0.0
    roc_auc = 0.0
    precision = 0.0
    recall = 0.0
    
    all_predictions = []
    
    with torch.no_grad():
        for X, y in test_data_loader:
            outputs = model(X)
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()
            
            # Store prediction history
            for i in range(len(X)):
                prob = probabilities[i].item()
                pred_class = int(predicted[i].item())
                confidence = prob if pred_class == 1 else (1 - prob)
                
                all_predictions.append({
                    "sample_number": len(all_predictions),
                    "probability": prob,
                    "predicted_class": pred_class,
                    "confidence": confidence,
                    "true_label": int(y[i].item())
                })
            
            accuracy += accuracy_score(y.numpy(), predicted.numpy())
            f1 += f1_score(y.numpy(), predicted.numpy())
            roc_auc += roc_auc_score(y.numpy(), predicted.numpy())
            precision += precision_score(y.numpy(), predicted.numpy())
            recall += recall_score(y.numpy(), predicted.numpy())
        
    accuracy /= len(test_data_loader)
    f1 /= len(test_data_loader)
    roc_auc /= len(test_data_loader)
    precision /= len(test_data_loader)
    recall /= len(test_data_loader)
    
    results = {
        "metrics": {
            "accuracy": accuracy,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
        },
        "total_samples": len(all_predictions),
        "predictions": all_predictions
    }
    
    return results
    
def evaluate_shap_values(model: BaseLoanNN, test_dataset: DataFrameDataset) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Calculate SHAP values for the trained model and generate visualization.
    
    Args:
        model: Trained neural network model
        test_dataset: Test dataset (used as background data for SHAP)
        
    Returns:
        Tuple of (shap_values_df, matplotlib figure)
    """
    model.eval()
    
    # Extract feature data and feature names from the dataset
    feature_columns = test_dataset.feature_columns
    
    # Get the X data as numpy array (shape: [n_samples, n_features])
    X_data = test_dataset.X.numpy()
    
    # Create a wrapper function for SHAP that returns numpy arrays
    def model_predict(x):
        model.eval()
        with torch.no_grad():
            tensor_input = torch.FloatTensor(x)
            output = model(tensor_input)
            # Return probability using sigmoid, reshape to 1D array
            probs = torch.sigmoid(output).numpy() # sigmoid -> 0.96 | softmax -> [0.04, 0.96]
            # Ensure output is always 1D (flatten if needed)
            return probs.flatten() # -> [...]
    
    # Use a subset of data as background for SHAP (for performance)
    background_size = min(100, len(X_data))
    background_data = X_data[:background_size]
    
    # Initialize SHAP explainer
    explainer = shap.KernelExplainer(model_predict, background_data)
    
    # Calculate SHAP values for a sample of data
    sample_size = min(200, len(X_data))
    sample_data = X_data[:sample_size]
    shap_values = explainer.shap_values(sample_data)
    
    # Create DataFrame with SHAP values
    shap_df = pd.DataFrame(
        shap_values,
        columns=feature_columns
    )
    
    # Generate SHAP summary plot
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        sample_data, 
        feature_names=feature_columns,
        show=False
    )
    plt.tight_layout()
    
    return shap_df, fig
    