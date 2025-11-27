from kedro.io import AbstractDataset
import torch
from pathlib import Path
from typing import Any

class PyTorchModel(AbstractDataset):
    def __init__(self, filepath: str):
        self._filepath = Path(filepath)
    
    def _load(self) -> Any:
        return torch.load(self._filepath, weights_only=False)
    
    def _save(self, model: Any) -> None:
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, self._filepath)
    
    def _describe(self):
        return dict(filepath=str(self._filepath))