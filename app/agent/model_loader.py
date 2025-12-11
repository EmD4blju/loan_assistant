import sys
from pathlib import Path

import torch
import torch.nn as nn
from .neural_network import BaseLoanNN
import joblib


class ModelLoader:

    @staticmethod
    def _load_model():
        model = torch.load(Path(__file__).parents[2] / 'models' / 'temp_scaled_loan_model.pth', weights_only=False)
        return model

    @staticmethod
    def _load_scaler():
        return joblib.load(Path(__file__).parents[2] / 'models' / 'feature_scaler.joblib')

    @staticmethod
    def _load_categorical_encoders():
        return joblib.load(Path(__file__).parents[2] / 'models' / 'feature_encoders.joblib')


if __name__ == "__main__":
    loader = ModelLoader()
    print(loader)
