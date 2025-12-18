"""Utility functions for model operations."""
import json
from pathlib import Path
from typing import Dict, Any
import torch
from torch import nn


def reconstruct_model_from_config(config: Dict[str, Any]):
    """Reconstruct a neural network model from configuration.
    
    Args:
        config: Dictionary containing model architecture configuration with keys:
            - input_size: Number of input features
            - hidden_layers: List of hidden layer sizes
            - output_size: Number of output features (default: 1)
            - is_temperature_scaled: Whether to wrap in TemperatureScaledNN (default: False)
    
    Returns:
        Reconstructed model (either BaseLoanNN or TemperatureScaledNN)
    """
    from loan_assistant.models.neural_network import BaseLoanNN, TemperatureScaledNN
    
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
    
    # Wrap with temperature scaling if needed
    if config.get('is_temperature_scaled', False):
        model = TemperatureScaledNN(base_model)
    else:
        model = base_model
    
    return model


def load_model_from_weights(weights_path: Path, config_path: Path):
    """Load a model by reconstructing from config and loading weights.
    
    Args:
        weights_path: Path to the model weights file (.pth)
        config_path: Path to the model configuration file (.json)
    
    Returns:
        Loaded model with weights
    """
    # Load model configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Reconstruct model architecture
    model = reconstruct_model_from_config(config)
    
    # Load weights
    state_dict = torch.load(weights_path, weights_only=True)
    model.load_state_dict(state_dict)
    
    return model


def extract_model_config(model):
    """Extract configuration from a model for saving.
    
    Args:
        model: Model instance (BaseLoanNN or TemperatureScaledNN)
    
    Returns:
        Dictionary containing model configuration
    
    Raises:
        ValueError: If model type is not supported
    """
    from loan_assistant.models.neural_network import BaseLoanNN, TemperatureScaledNN
    
    # Determine if this is a temperature-scaled model
    is_temperature_scaled = isinstance(model, TemperatureScaledNN)
    
    if is_temperature_scaled:
        base_model = model.model
    else:
        base_model = model
    
    # Extract architecture configuration from the base model
    if isinstance(base_model, BaseLoanNN):
        # Get input size from first hidden layer or output layer
        if len(base_model.hidden_layers) > 0:
            input_size = base_model.hidden_layers[0].in_features
            hidden_layer_sizes = [layer.out_features for layer in base_model.hidden_layers]
        else:
            input_size = base_model.output_layer.in_features
            hidden_layer_sizes = []
        
        output_size = base_model.output_layer.out_features
        
        config = {
            'input_size': input_size,
            'hidden_layers': hidden_layer_sizes,
            'output_size': output_size,
            'is_temperature_scaled': is_temperature_scaled
        }
        
        return config
    else:
        raise ValueError(f"Unsupported model type: {type(base_model)}")
