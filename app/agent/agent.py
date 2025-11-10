import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
    
import torch
import torch.nn as nn
from neural_core.models.neural_network import BaseLoanNN
import joblib

model = BaseLoanNN(input_size=13, hidden_layers=nn.ModuleList([nn.Linear(13, 64)]), output_size=1)
model.load_state_dict(torch.load(Path(__file__).parent / 'modules' / 'model.pth'))
scaler = joblib.load(Path(__file__).parent / 'modules' / 'numeric_scaler.joblib')
categorical_encoders = joblib.load(Path(__file__).parent / 'modules' / 'categorical_encoders.joblib')

print(model)
print(scaler)
print(categorical_encoders)