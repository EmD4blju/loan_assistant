import sys
from pathlib import Path
import torch
from models.arena import Arena
from models.dataset import DataFrameDataset
from models.neural_network import BaseLoanNN
from torch.optim import Adam
import torch.nn as nn
from logging import getLogger
from tools.log_controller import LogController

log = getLogger(__name__)

def main():
    log_controller = LogController(config_path=Path(__file__).parent / 'config' / 'logging_config.json', dirs=Path(__file__).parent / 'logs')
    log_controller.start()
    
    log.info("Welcome to Loan Assistant Neural Core!")
    
    #~ Load dataset
    dataset_path = Path(__file__).parent / 'repo' / 'loan_data.csv'
    log.info(f"Loading dataset from: {dataset_path}")
    dataset = DataFrameDataset.from_csv(dataset_path, target_column='loan_status')
    log.info("Dataset loaded successfully.")
    
    #~ Hyperparameter tuning
    log.info("Starting hyperparameter tuning...")
    def objective(trial):
            input_size = len(dataset.feature_columns)
            hidden_size_1 = trial.suggest_int('hidden_size_1', 32, 128)
            hidden_size_2 = trial.suggest_int('hidden_size_2', 32, 128)
            lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
            
            model = BaseLoanNN(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2, output_size=1)
            optimizer = Adam(model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()

            epochs = 30
            arena = Arena(model, optimizer, criterion, dataset, enable_validation=True)
            arena.train(epochs=epochs, batch_size=32)
            
            return arena.average_val_bce
        
    best_params, best_value = Arena.tune_hyperparameters(
        objective=objective,
        n_trials=20
    )
    log.info("Hyperparameter tuning finished.")

    #~ Initialize model, optimizer, and criterion with best params
    log.info("Initializing model with best hyperparameters...")
    input_size = len(dataset.feature_columns)
    model = BaseLoanNN(input_size=input_size, hidden_size_1=best_params['hidden_size_1'], hidden_size_2=best_params['hidden_size_2'], output_size=1)
    optimizer = Adam(model.parameters(), lr=best_params['lr'])
    criterion = torch.nn.BCEWithLogitsLoss()
    log.info("Model initialized.")
    
    #~ Initialize Arena with the best model
    arena = Arena(model, optimizer, criterion, dataset, enable_validation=True)
    
    #~ Train the final model
    log.info("Starting final training with best hyperparameters...")
    arena.train(epochs=30, batch_size=32)
    log.info("Training finished.")
    
    #~ Evaluate the model
    log.info("Starting evaluation...")
    arena.evaluate(batch_size=32)
    log.info("Evaluation finished.")
    
    #~ Save results to report.txt
    report_path = Path(__file__).parent / 'logs' / 'report.txt'
    log.info(f"Saving report to: {report_path}")
    with open(report_path, 'a') as f:
        f.write("\n=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")  # Separate from previous runs
        f.write("Best Hyperparameters:\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"Best Validation BCE: {best_value:.4f}\n\n")
        f.write("Final Evaluation Results:\n")
        f.write(arena.results)
        f.write("\n=-=-=-=-=-=-=-=-=-=-=-=-=-=\n") # End separator
        
    log.info(f"Report saved successfully.")

if __name__ == "__main__":
    main()