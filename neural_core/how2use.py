import torch
from torch import nn
from pathlib import Path
from models.neural_network import BaseLoanNN
from models.dataset import DataFrameDataset

def load_model(model_path:Path, input_size:int, hidden_layers:nn.ModuleList, output_size:int) -> nn.Module:
    """Loads a trained model from the specified path.

    Args:
        model_path (Path): Path to the saved model file.
        input_size (int): Size of the input layer.
        hidden_layers (nn.ModuleList): List of hidden layers.
        output_size (int): Size of the output layer.

    Returns:
        nn.Module: Loaded model.
    """
    model = BaseLoanNN(input_size=input_size, hidden_layers=hidden_layers, output_size=output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


if __name__ == "__main__":
    model = load_model(Path('neural_core/model.pth'), input_size=13, hidden_layers=nn.ModuleList([nn.Linear(13, 64)]), output_size=1)
    print("Model loaded successfully.")
    # Get some sample data
    dataset = DataFrameDataset.from_csv(Path(__file__).parent / 'repo' / 'loan_data.csv', target_column='loan_status')
    # Do prediction
    predictions = model(dataset.features[:5])
    correct = dataset.target[:5]
    # Apply sigmoid to get probabilities
    probabilities = torch.sigmoid(predictions)
    
    percentages = probabilities * 100

    print("Predicted % probabilities for the first 5 samples:")
    for i, p in enumerate(percentages):
        print(f"Sample {i+1}: {p.item():.2f}%  (True label: {int(correct[i].item())})")
        
    # The output will show the predicted probabilities along with the true labels for the first 5 samples in the dataset.
    # Note that the sigmoid function is applied to convert logits to probabilities of the correct class to be 1.
    # So if the true label is 1, a higher probability indicates a better prediction.
    # On the other hand, if the true label is 0, a lower probability indicates a better prediction.
    # The model can be used for now, If I happen to develop any better we could just swap it out later.
    
    