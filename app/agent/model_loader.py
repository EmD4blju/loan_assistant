import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import torch
import torch.nn as nn
from neural_core.models.neural_network import BaseLoanNN
import joblib


class ModelLoader:

    @staticmethod
    def _load_model():
        model = BaseLoanNN(input_size=13, hidden_layers=nn.ModuleList([nn.Linear(13, 64)]), output_size=1)
        model.load_state_dict(torch.load(Path(__file__).parent / 'modules' / 'model.pth'))
        return model

    @staticmethod
    def _load_scaler():
        return joblib.load(Path(__file__).parent / 'modules' / 'numeric_scaler.joblib')

    @staticmethod
    def _load_categorical_encoders():
        return joblib.load(Path(__file__).parent / 'modules' / 'categorical_encoders.joblib')


if __name__ == "__main__":
    loader = ModelLoader()
    print(loader)
