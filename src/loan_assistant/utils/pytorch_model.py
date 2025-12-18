from kedro.io import AbstractDataset
import torch
import json
from pathlib import Path
from typing import Any
from torch import nn

class PyTorchModel(AbstractDataset):
    def __init__(self, filepath: str):
        self._filepath = Path(filepath)
        self._config_filepath = self._filepath.parent / f"{self._filepath.stem}_config.json"
    
    def _load(self) -> Any:
        """Load model by reconstructing architecture from config and loading weights."""
        # Import here to avoid circular dependencies
        from loan_assistant.models.neural_network import BaseLoanNN, TemperatureScaledNN
        
        # Load model configuration
        with open(self._config_filepath, 'r') as f:
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
        state_dict = torch.load(self._filepath, weights_only=True)
        model.load_state_dict(state_dict)
        
        return model
    
    def _save(self, model: Any) -> None:
        """Save model weights and architecture configuration separately."""
        # Import here to avoid circular dependencies
        from loan_assistant.models.neural_network import BaseLoanNN, TemperatureScaledNN
        
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        
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
            
            # Save configuration
            with open(self._config_filepath, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported model type: {type(base_model)}")
        
        # Save only the weights
        torch.save(model.state_dict(), self._filepath)
    
    def _describe(self):
        return dict(
            filepath=str(self._filepath),
            config_filepath=str(self._config_filepath)
        )