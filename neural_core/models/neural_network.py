import torch.nn as nn
from torch.nn import functional as F
from torch import softmax
from typing import override


class BaseLoanNN(nn.Module):
    """ A base neural network class for loan prediction tasks.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layer_1 = nn.Linear(13, 50)  # Example layers
        self.layer_2 = nn.Linear(50, 20)   # Example layers
        self.output_layer = nn.Linear(20, 1)  # Example layers
    
    @override
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.output_layer(x)
        return x
    
    @override
    def __str__(self):
        return f'BaseLoanNN [{self.layer_1}, {self.layer_2}, {self.output_layer}]'