import pandas as pd
from torch.utils.data import Dataset
from torch import tensor
import torch
import pandas as pd
from logging import getLogger

log = getLogger(__name__)

class DataFrameDataset(Dataset):
    """ A PyTorch Dataset class for loading data from a pandas DataFrame.
    """
    def __init__(self, df:pd.DataFrame, target_column:str=None, clean:bool=True):
        self.df = df.copy()
        self.target_column = target_column if target_column in self.df.columns else self.df.columns[-1]
        self.feature_columns = [column for column in self.df.columns if column != self.target_column]
        self.X = tensor(self.df.drop(self.target_column, axis=1).to_numpy(), dtype=torch.float32)
        self.y = tensor(self.df[self.target_column].to_numpy(), dtype=torch.float32)
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx:int):
        return self.X[idx], self.y[idx]
    
    def __str__(self):
        return f'DataFrameDataset with {len(self)} samples, {len(self.feature_columns)} features, target column: {self.target_column}'