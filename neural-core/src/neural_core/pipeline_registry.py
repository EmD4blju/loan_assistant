"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline
from neural_core.pipelines.preprocessing.pipeline import create_preprocessing_pipeline
from neural_core.pipelines.tuning.pipeline import create_tuning_pipeline
from neural_core.pipelines.modelling.pipeline import create_modelling_pipeline
from neural_core.pipelines.evaluation.pipeline import create_evaluation_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    preprocessing_pipeline = create_preprocessing_pipeline()
    tuning_pipeline = create_tuning_pipeline()
    modelling_pipeline = create_modelling_pipeline()
    evaluation_pipeline = create_evaluation_pipeline()

    return {
        "preprocessing": preprocessing_pipeline,
        "tuning": tuning_pipeline,
        "modelling": modelling_pipeline,
        "evaluation": evaluation_pipeline,
        "__default__": evaluation_pipeline,
    }
    
