import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class DataFrameDataset(Dataset):
    def __init__(self, dataframe, target_column=None):
        self.df = dataframe
        if target_column is None:
            target_column = list(self.df.columns)[-1]
        self.categorical_translations = dict()
        not_float_columns = set(self.df.select_dtypes(include=['object']).columns)
        for name in not_float_columns:
            self.df[name] = self._translate_categorical(name)
        self.features = self.df.drop(target_column, axis=1).values.astype(np.float32)
        self.target = self.df[target_column].values.astype(np.float32)
        self.length = self.df.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

    def _translate_categorical(self, column_name):
        categorical_column = self.df[column_name]
        values = list(set(categorical_column))
        translations = {values[i]: i for i in range(len(values))}
        translated = pd.DataFrame([translations[v] for i, v in categorical_column.items()]).values.astype(np.float32)
        self.categorical_translations[column_name] = {v: k for k, v in translations.items()}
        return translated

    @staticmethod
    def from_csv(path):
        return DataFrameDataset(pd.read_csv(path))

#usage: dataset = DataFrameDataset.from_csv('<<path to csv>>')