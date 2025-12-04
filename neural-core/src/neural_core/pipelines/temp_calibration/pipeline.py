from kedro.pipeline import Pipeline, node
from .nodes import prepare_data, train_temperature_scaling_model

def create_temp_calibration_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=prepare_data,
                inputs="val_data",
                outputs="val_dataset",
                name="prepare_temp_calibration_data_node",
            ),
            node(
                func=train_temperature_scaling_model,
                inputs=["trained_model", "val_dataset"],
                outputs="temp_scaled_model",
                name="train_temperature_scaling_model_node",
            ),
        ]
    )