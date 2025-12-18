import sys
from pathlib import Path
import joblib

# Add src to path to import model utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from loan_assistant.models.model_utils import load_model_from_weights

# Define the absolute path to the project root to locate model files
PROJECT_ROOT = Path(__file__).parent.parent.parent

class ModelLoader:

    @staticmethod
    def _load_model():
        """Load model by reconstructing architecture from config and loading weights."""
        model_path = PROJECT_ROOT / 'models' / 'temp_scaled_loan_model.pth'
        config_path = PROJECT_ROOT / 'models' / 'temp_scaled_loan_model_config.json'
        return load_model_from_weights(model_path, config_path)

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
