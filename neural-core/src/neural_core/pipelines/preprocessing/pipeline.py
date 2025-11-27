from kedro.pipeline import Pipeline, node
from .nodes import clean_data, encode_categorical, scale_numerical, split_data


def create_preprocessing_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=clean_data,
                inputs="raw_data",
                outputs="cleaned_data",
                name="clean_data_node",
            ),
            node(
                func=encode_categorical,
                inputs="cleaned_data",
                outputs=["encoded_data", "encoders"],
            ),
            node(
                func=scale_numerical,
                inputs=["encoded_data", "params:scaling_method"],
                outputs=["scaled_data", "scaler"],
                name="scale_data_node",
            ),
            node(
                func=split_data,
                inputs=["scaled_data", "params:val_size", "params:test_size"],
                outputs=["train_data", "val_data", "test_data"],
                name="split_data_node",
            ),
        ]
    )