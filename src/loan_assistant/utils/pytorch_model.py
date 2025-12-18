from kedro.io import AbstractDataset
import torch
import json
from pathlib import Path
from typing import Any

class PyTorchModel(AbstractDataset):
    def __init__(self, filepath: str):
        self._filepath = Path(filepath)
        self._config_filepath = self._filepath.parent / f"{self._filepath.stem}_config.json"
    
    def _load(self) -> Any:
        """Load model by reconstructing architecture from config and loading weights."""
        from loan_assistant.models.model_utils import load_model_from_weights
        return load_model_from_weights(self._filepath, self._config_filepath)
    
    def _save(self, model: Any) -> None:
        """Save model weights and architecture configuration separately."""
        from loan_assistant.models.model_utils import extract_model_config
        
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract and save configuration
        config = extract_model_config(model)
        with open(self._config_filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save only the weights
        torch.save(model.state_dict(), self._filepath)
    
    def _describe(self):
        return dict(
            filepath=str(self._filepath),
            config_filepath=str(self._config_filepath)
        )