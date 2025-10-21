import torch.nn as nn
from torch.nn import functional as F
from torch import softmax
from typing import override


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
    
    
    @override
    def __str__(self):
        return f'BaseLoanNN [{self.layer_1}, {self.layer_2}, {self.output_layer}]'