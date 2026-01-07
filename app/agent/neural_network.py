import torch.nn as nn
from torch.nn import functional as F
import torch
from typing import override
from logging import getLogger

log = getLogger(__name__)

class BaseLoanNN(nn.Module):
    """ A base neural network class for loan prediction tasks.
    """
    def __init__(self, input_size: int, hidden_layers: nn.ModuleList, output_size: int = 1):
        super().__init__()
        log.debug(f"Initializing BaseLoanNN with input_size={input_size}, hidden_layers={hidden_layers}, output_size={output_size}")
        
        self.hidden_layers = hidden_layers
        
        if len(self.hidden_layers) > 0:
            # The input of the output layer is the output of the last hidden layer
            output_layer_input_dim = self.hidden_layers[-1].out_features
        else:
            # No hidden layers, output layer takes the main input
            output_layer_input_dim = input_size
            
        self.output_layer = nn.Linear(output_layer_input_dim, output_size)
    
    @override
    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x
        
    @override
    def __str__(self):
        layers = [str(layer) for layer in self.hidden_layers] + [str(self.output_layer)]
        return f'BaseLoanNN [{", ".join(layers)}]'
    

class TemperatureScaledNN(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Initial temperature value
    
    @override
    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature
    
    def calibrate_temperature(self, validation_loader):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        logits_list = []
        labels_list = []
        
        self.model.eval()
        with torch.no_grad():
            for X, y in validation_loader:
                logits = self.model(X)
                logits_list.append(logits)
                labels_list.append(y)
        logits = torch.cat(logits_list).squeeze()
        labels = torch.cat(labels_list).squeeze()
        
        def eval():
            optimizer.zero_grad()
            loss = criterion(logits / self.temperature, labels)
            loss.backward()
            with torch.no_grad():
                self.temperature.clamp_(min=0.1, max=10.0)
            return loss
        
        optimizer.step(eval)
        log.info(f'Optimal temperature: {self.temperature.item():.4f}')
        
        return self.temperature.item()
        
        
        
        
