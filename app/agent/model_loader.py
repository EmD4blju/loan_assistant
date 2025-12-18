import sys
from pathlib import Path
import torch
import json
import joblib
from torch import nn

# Import the neural network classes
from app.agent.neural_network import BaseLoanNN, TemperatureScaledNN

# Define the absolute path to the project root to locate model files
PROJECT_ROOT = Path(__file__).parent.parent.parent

class ModelLoader:

    @staticmethod
    def _load_model():
        """Load model by reconstructing architecture from config and loading weights."""
        model_path = PROJECT_ROOT / 'models' / 'temp_scaled_loan_model.pth'
        config_path = PROJECT_ROOT / 'models' / 'temp_scaled_loan_model_config.json'
        
        # Load model configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Reconstruct the base model architecture
        input_size = config['input_size']
        hidden_layer_sizes = config['hidden_layers']
        output_size = config.get('output_size', 1)
        
        # Build hidden layers
        hidden_layers = nn.ModuleList()
        last_dim = input_size
        for hidden_dim in hidden_layer_sizes:
            hidden_layers.append(nn.Linear(last_dim, hidden_dim))
            last_dim = hidden_dim
        
        # Create base model
        base_model = BaseLoanNN(
            input_size=input_size,
            hidden_layers=hidden_layers,
            output_size=output_size
        )
        
        # Check if this is a temperature-scaled model
        if config.get('is_temperature_scaled', False):
            model = TemperatureScaledNN(base_model)
        else:
            model = base_model
        
        # Load weights
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict)
        
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
