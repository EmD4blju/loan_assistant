from kedro.pipeline import Pipeline, node
from .nodes import evaluate_model, prepare_data, evaluate_shap_values

def create_evaluation_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=prepare_data,
                inputs=["train_data", "test_data"],
                outputs=["train_dataset", "test_dataset"],
                name="prepare_evaluation_data_node",
            ),
            node(
                func=evaluate_model,
                inputs=["trained_model", "test_dataset"],
                outputs="evaluation_results",
                name="evaluate_model_node",
            ),
            node(
                func=evaluate_shap_values,
                inputs=["trained_model", "test_dataset"],
                outputs=["shap_values", "shap_plot"],
                name="evaluate_shap_values_node",
            ),
        ]
    )