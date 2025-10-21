import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch import tensor
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from pathlib import Path


class DataFrameDataset(Dataset):
    """ A PyTorch Dataset class for loading data from a pandas DataFrame.
    """
    def __init__(self, dataframe:pd.DataFrame, target_column:str=None, clean:bool=True):
        self.df = dataframe if not clean else self._clean_data(dataframe)
        self.target_column = target_column if target_column in self.df.columns else self.df.columns[-1]
        self.feature_columns = [column for column in self.df.columns if column != self.target_column]
        self.features = tensor(self.df.drop(self.target_column, axis=1).to_numpy(), dtype=torch.float32)
        self.target = tensor(self.df[self.target_column].to_numpy(), dtype=torch.float32)
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx:int):
        return self.features[idx], self.target[idx]
    
    def __str__(self):
        return f'DataFrameDataset with {len(self)} samples, {len(self.feature_columns)} features, target column: {self.target_column}'

    @staticmethod
    def from_csv(path:Path, target_column:str=None, clean:bool=True) -> 'DataFrameDataset':
        """ Creates a DataFrameDataset from a CSV file.

        Args:
            path (Path): Path to the CSV file.
            target_column (str, optional): Name of the target column. If None, the last column is used. Defaults to None.
            clean (bool, optional): Whether to clean the data. Defaults to True.

        Returns:
            DataFrameDataset: Instance of DataFrameDataset.
        """
        return DataFrameDataset(pd.read_csv(path), target_column, clean)
    
    def _clean_data(self, df:pd.DataFrame) -> pd.DataFrame:
        """ Cleans the dataframe by performing the following operations:
        1. Undersampling to balance the classes in the 'loan_status' column.
        2. Encoding categorical variables using Label Encoding.
        3. Removing anomalies in the 'person_age' column using the IQR method.

        Args:
            df (pd.DataFrame): Input dataframe to be cleaned.

        Returns:
            pd.DataFrame: Cleaned dataframe.
        """
        #~ Undersampling by loan status
        df_yes = df[df['loan_status'] == 1]
        df_no = df[df['loan_status'] == 0]
        df_no_sampled = df_no.sample(n=len(df_yes))
        df_balanced = pd.concat([df_yes, df_no_sampled])
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        df = df_balanced
        
        #~ Remove anomalies
        Q1 = df['person_age'].quantile(0.25)
        Q3 = df['person_age'].quantile(0.975) # Using 97.5 percentile to retain more data (up to 70 years old)
        IQR = Q3 - Q1
        df = df[(df['person_age'] >= Q1 - 1.5 * IQR) & (df['person_age'] <= Q3 + 1.5 * IQR)].copy()
        
        #~ Encode categorical variables
        encoder = LabelEncoder()
        categorical_features = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']
        for feature in categorical_features:
            df[feature] = encoder.fit_transform(df[feature])
        
        #~ Standardize numerical features
        numeric_features = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'person_emp_exp', 'credit_score']
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
        
        return df