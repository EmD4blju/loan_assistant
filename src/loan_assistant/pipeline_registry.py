"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline
from loan_assistant.pipelines.preprocessing.pipeline import create_preprocessing_pipeline
from loan_assistant.pipelines.tuning.pipeline import create_tuning_pipeline
from loan_assistant.pipelines.modelling.pipeline import create_modelling_pipeline
from loan_assistant.pipelines.evaluation.pipeline import create_evaluation_pipeline
from loan_assistant.pipelines.temp_calibration.pipeline import create_temp_calibration_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    preprocessing_pipeline = create_preprocessing_pipeline()
    tuning_pipeline = create_tuning_pipeline()
    modelling_pipeline = create_modelling_pipeline()
    temp_calibration_pipeline = create_temp_calibration_pipeline()
    evaluation_pipeline = create_evaluation_pipeline()
    

    return {
        "preprocessing": preprocessing_pipeline,
        "tuning": tuning_pipeline,
        "modelling": modelling_pipeline,
        "temp_calibration": temp_calibration_pipeline,
        "evaluation": evaluation_pipeline,
        "__default__": evaluation_pipeline,
    }
    
