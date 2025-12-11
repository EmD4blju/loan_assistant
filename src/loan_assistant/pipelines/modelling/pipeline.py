from kedro.pipeline import Pipeline, node
from .nodes import prepare_data, train_model

def create_modelling_pipeline() -> Pipeline:
    return Pipeline([
        node(
            func=prepare_data,
            inputs=["train_data", "val_data"],
            outputs=["train_dataset", "val_dataset"],
            name="prepare_data_node",
        ),
        node(
            func=train_model,
            inputs=["train_dataset", "val_dataset", "params:hidden_layers", "params:learning_rate", "params:epochs"],
            outputs="trained_model",
            name="train_model_node",
        ),
    ])