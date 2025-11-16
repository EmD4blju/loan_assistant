import pandas as pd
from neural_core.models.neural_network import BaseLoanNN
import torch
from torch.utils.data import Dataset, DataLoader
from neural_core.models.dataset import DataFrameDataset
from typing import Tuple
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score


def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[Dataset, Dataset]:
    train_dataset = DataFrameDataset(train_df, target_column='loan_status')
    test_dataset = DataFrameDataset(test_df, target_column='loan_status')
    return train_dataset, test_dataset

def evaluate_model(model: BaseLoanNN, test_dataset: Dataset) -> pd.DataFrame:
    test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model.eval()
    
    accuracy = 0.0
    f1 = 0.0
    roc_auc = 0.0
    precision = 0.0
    recall = 0.0
    
    
    with torch.no_grad():
        for X, y in test_data_loader:
            outputs = model(X)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            
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
    
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall
    }