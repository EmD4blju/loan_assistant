import pandas as pd
import optuna
from loan_assistant.models.neural_network import BaseLoanNN
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from loan_assistant.models.dataset import DataFrameDataset
from typing import Tuple


def prepare_data(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[Dataset, Dataset]:
    train_dataset = DataFrameDataset(train_df, target_column='loan_status')
    val_dataset = DataFrameDataset(val_df, target_column='loan_status')
    return train_dataset, val_dataset


def tune_hyperparameters(train_dataset: Dataset, val_dataset: Dataset, n_trials: int = 20):
    def objective(trial):
            input_size = len(train_dataset.feature_columns)
            num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 5)
            
            hidden_layers = nn.ModuleList()
            last_dim = input_size
            for i in range(num_hidden_layers):
                hidden_dim = trial.suggest_int(f'hidden_dim_{i}', 32, 128)
                hidden_layers.append(nn.Linear(last_dim, hidden_dim))
                last_dim = hidden_dim
            
            lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
            epochs = trial.suggest_int('epochs', 10, 50)
            
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
                        
            val_bce = val_loss / (len(val_data_loader))
                
            return val_bce
    
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    return {"params": study.best_trial.params, "value": study.best_trial.value}





    