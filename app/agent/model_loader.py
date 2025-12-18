import sys
from pathlib import Path

import sys
from pathlib import Path
import torch
import joblib

# Explicitly import the class required by torch.load
# This makes the class definition available in the current scope.
from app.agent.neural_network import BaseLoanNN

# Define the absolute path to the project root to locate model files
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Define the absolute path to the project root to locate model files
PROJECT_ROOT = Path(__file__).parent.parent.parent

class ModelLoader:

    @staticmethod
    def _load_model():
        model_path = PROJECT_ROOT / 'models' / 'temp_scaled_loan_model.pth'
        model = torch.load(model_path, weights_only=False)
        return model

    @staticmethod
    def _load_scaler():
        scaler_path = PROJECT_ROOT / 'models' / 'feature_scaler.joblib'
        return joblib.load(scaler_path)

    @staticmethod
    def _load_categorical_encoders():
        encoders_path = PROJECT_ROOT / 'models' / 'feature_encoders.joblib'
        return joblib.load(encoders_path)


if __name__ == "__main__":
    loader = ModelLoader()
    print(loader)
