import pandas as pd
from neural_core.models.neural_network import BaseLoanNN
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from neural_core.models.dataset import DataFrameDataset
from typing import Tuple, List

def prepare_data(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[Dataset, Dataset]:
    train_dataset = DataFrameDataset(train_df, target_column='loan_status')
    val_dataset = DataFrameDataset(val_df, target_column='loan_status')
    return train_dataset, val_dataset

def train_model(train_dataset: Dataset, val_dataset: Dataset, hidden_layers_sizes: List[int], lr: float = 1e-3, epochs: int = 20) -> BaseLoanNN:
    input_size = len(train_dataset.feature_columns)
    
    hidden_layers = nn.ModuleList()
    last_dim = input_size
    for hidden_dim in hidden_layers_sizes:
        hidden_layers.append(nn.Linear(last_dim, hidden_dim))
        last_dim = hidden_dim

    model = BaseLoanNN(input_size=input_size, hidden_layers=hidden_layers, output_size=1)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        for X, y in train_data_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y.reshape(-1, 1))
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_data_loader:
                outputs = model(X)
                loss = criterion(outputs, y.reshape(-1, 1))
                val_loss += loss.item()
            val_loss /= len(val_data_loader)
            if val_loss < 0.1:
                break  # Early stopping criterion    
                
    return model