import pandas as pd
from loan_assistant.models.dataset import DataFrameDataset
from loan_assistant.models.neural_network import BaseLoanNN, TemperatureScaledNN
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


def prepare_data(val_df: pd.DataFrame) -> Tuple[Dataset, Dataset]:
    val_dataset = DataFrameDataset(val_df, target_column='loan_status')
    return val_dataset
    
def train_temperature_scaling_model(model: BaseLoanNN, val_dataset: Dataset) -> TemperatureScaledNN:
    val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    temp_scaled_model = TemperatureScaledNN(model)
    temp_scaled_model.calibrate_temperature(val_data_loader)
    return temp_scaled_model