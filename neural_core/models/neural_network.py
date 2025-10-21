import torch.nn as nn
from torch.nn import functional as F
from torch import softmax
from torch.utils.data import DataLoader, Dataset
from typing import override
import pandas as pd
import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn.model_selection import train_test_split

class BaseLoanNN(nn.Module):
    """ A base neural network class for loan prediction tasks.
    """
    def __init__(self, input_size: int = 13, hidden_size: int = 50, output_size: int = 1):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    @override
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.output_layer(x)
        return x
    
    
    def perform_training(self, train_dataset:Dataset, optimizer, criterion ,epochs:int=50, batch_size:int=32, enable_validation:bool = False) -> None:
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if enable_validation:
            train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for X, y in data_loader:
                optimizer.zero_grad()
                outputs = self(X)
                loss = criterion(outputs, y.reshape(-1, 1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            print(f'Epoch [{epoch+1}/{epochs}], BCE: {epoch_loss/len(data_loader):.4f}')
            
            if enable_validation:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_val, y_val in val_loader:
                        val_outputs = self(X_val)
                        v_loss = criterion(val_outputs, y_val.reshape(-1, 1))
                        val_loss += v_loss.item()
                print(f'Validation BCE: {val_loss/len(val_loader):.4f}')
        
        
    
    @override
    def __str__(self):
        return f'BaseLoanNN [{self.layer_1}, {self.layer_2}, {self.output_layer}]'