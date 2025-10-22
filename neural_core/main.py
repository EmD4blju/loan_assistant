import sys
from pathlib import Path
import torch
from models.arena import Arena
from models.dataset import DataFrameDataset
from models.neural_network import BaseLoanNN
from torch.optim import Adam

def main():
    print("Welcome to Loan Assistant Neural Core!")
    
    #~ Load dataset
    dataset_path = Path(__file__).parent / 'repo' / 'loan_data.csv'
    dataset = DataFrameDataset.from_csv(dataset_path, target_column='loan_status')
    
    #~ Hyperparameter tuning
    print("Starting hyperparameter tuning...")
    best_params, best_value = Arena.tune_hyperparameters(
        model_class=BaseLoanNN,
        dataset=dataset,
        n_trials=20,  
        epochs=30
    )
    print("Hyperparameter tuning finished.")

    #~ Initialize model, optimizer, and criterion with best params
    input_size = len(dataset.feature_columns)
    model = BaseLoanNN(input_size=input_size, hidden_size=best_params['hidden_size'])
    optimizer = Adam(model.parameters(), lr=best_params['lr'])
    criterion = torch.nn.BCEWithLogitsLoss()
    
    #~ Initialize Arena with the best model
    arena = Arena(model, optimizer, criterion, dataset, enable_validation=True)
    
    #~ Train the final model
    print("\nStarting final training with best hyperparameters...")
    arena.train(epochs=30, batch_size=best_params['batch_size'])
    print("Training finished.")
    
    #~ Evaluate the model
    print("Starting evaluation...")
    arena.evaluate(batch_size=best_params['batch_size'])
    print("Evaluation finished.")
    
    #~ Save results to report.txt
    report_path = Path(__file__).parent / 'report.txt'
    with open(report_path, 'w') as f:
        f.write("Best Hyperparameters:\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"Best Validation BCE: {best_value:.4f}\n\n")
        f.write("Final Evaluation Results:\n")
        f.write(arena.results)
        
    print(f"\nReport saved to {report_path}")
    print("\nFinal Results:")
    print(arena.results)

if __name__ == "__main__":
    main()