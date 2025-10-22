import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import optuna
from .neural_network import BaseLoanNN
from logging import getLogger

log = getLogger(__name__)

class Arena():
    def __init__(self, model:nn.Module, optimizer:torch.optim.Optimizer, criterion:nn.Module, dataset:Dataset, enable_validation:bool=True, test_split:float=0.2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f'Using device: {self.device}')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.enable_validation = enable_validation
        self._split_dataset(dataset, test_split)
        self.average_train_bce = 0.0
        self.average_val_bce = 0.0 if enable_validation else None
        self.average_test_bce = 0.0
        
    @property
    def results(self) -> str:
        res = f'Average Train BCE: {self.average_train_bce:.4f}\n'
        if self.enable_validation:
            res += f'Average Validation BCE: {self.average_val_bce:.4f}\n'
        res += f'Average Test BCE: {self.average_test_bce:.4f}\n'
        return res
            
    def _split_dataset(self, dataset:Dataset, test_size:float):
        self.train_dataset, self.test_dataset = train_test_split(dataset, test_size=test_size)
        if self.enable_validation:
            self.train_dataset, self.val_dataset = train_test_split(self.train_dataset, test_size=test_size)
        else:
            self.val_dataset = None
        
    def train(self, epochs:int=50, batch_size:int=32) -> None:
        train_data_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for X, y in train_data_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y.reshape(-1, 1))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                self.average_train_bce += loss.item()
            
            log.info(f'Epoch [{epoch+1}/{epochs}], BCE: {train_loss/len(train_data_loader):.4f}')
            
            if self.enable_validation:
                test_data_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X, y in test_data_loader:
                        X, y = X.to(self.device), y.to(self.device)
                        outputs = self.model(X)
                        loss = self.criterion(outputs, y.reshape(-1, 1))
                        val_loss += loss.item()
                        self.average_val_bce += loss.item()
                log.info(f'Validation BCE: {val_loss/len(test_data_loader):.4f}')
            
        self.average_train_bce /= epochs * len(train_data_loader)
        if self.enable_validation:
            self.average_val_bce /= epochs * len(test_data_loader)
    
    def evaluate(self, batch_size:int=32) -> None:
        data_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = self.criterion(outputs, y.reshape(-1, 1))
                total_loss += loss.item()
                self.average_test_bce += loss.item()
        self.average_test_bce /= len(data_loader)
        
    @staticmethod
    def tune_hyperparameters(objective, n_trials: int = 50):
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        log.info("Best trial:")
        trial = study.best_trial
        log.info(f"  Value: {trial.value}")
        log.info("  Params: ")
        for key, value in trial.params.items():
            log.info(f"    {key}: {value}")

        return trial.params, trial.value