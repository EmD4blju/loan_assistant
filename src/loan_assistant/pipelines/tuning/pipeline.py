from kedro.pipeline import Pipeline, node
from loan_assistant.pipelines.tuning.nodes import prepare_data, tune_hyperparameters

def create_tuning_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=prepare_data,
                inputs=["train_data", "val_data"],
                outputs=["train_dataset", "val_dataset"],
                name="prepare_data_node",
            ),
            node(
                func=tune_hyperparameters,
                inputs=["train_dataset", "val_dataset", "params:tuning_trials"],
                outputs="best_settings",
                name="tune_hyperparameters_node",
            ),
        ]
    )