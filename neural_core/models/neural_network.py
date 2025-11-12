import torch.nn as nn
from torch.nn import functional as F
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
            last_hidden_layer = self.hidden_layers[-1]
            if isinstance(last_hidden_layer, nn.Linear):
                output_layer_input_dim = last_hidden_layer.out_features
            else:
                # Fallback for non-linear layers, though we expect Linear
                # This might need adjustment based on layer types used
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