import json
import logging
import os
import time
from typing import Any

import mlflow


DEFAULT_EXPERIMENT_NAME = "newstalk"
DEFAULT_TRACKING_URI = "http://localhost:5000"
_DSPY_AUTOLOG_ENABLED = False


def configure_mlflow(
    tracking_uri: str | None = None, experiment_name: str | None = None
) -> None:
    """Configure MLflow tracking URI and experiment name."""
    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    if not tracking_uri.startswith(("http://", "https://")):
        logging.warning(
            "MLFLOW_TRACKING_URI points to a local path (%s); using %s instead.",
            tracking_uri,
            DEFAULT_TRACKING_URI,
        )
        tracking_uri = DEFAULT_TRACKING_URI
    if experiment_name is None:
        experiment_name = os.environ.get(
            "MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME
        )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logging.info(
        "MLflow configured tracking_uri=%s experiment=%s", tracking_uri, experiment_name
    )


def configure_dspy_autolog(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> None:
    """Configure MLflow and enable DSPy autolog tracing."""
    global _DSPY_AUTOLOG_ENABLED
    configure_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)
    if _DSPY_AUTOLOG_ENABLED:
        return
    logging.info("MLflow DSPy autolog enabling")
    started_at = time.perf_counter()
    mlflow.dspy.autolog()
    elapsed = time.perf_counter() - started_at
    _DSPY_AUTOLOG_ENABLED = True
    logging.info("MLflow DSPy autolog enabled seconds=%.2f", elapsed)


def _stringify_tag_value(value: Any) -> str:
    """Return a string-safe MLflow tag value."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def log_inference_call(
    name: str,
    model: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    metadata: dict[str, Any],
    duration_seconds: float,
) -> None:
    """Log a single inference call with inputs and outputs to MLflow."""
    configure_mlflow()
    logging.info(
        "MLflow logging inference name=%s model=%s duration_seconds=%.2f",
        name,
        model,
        duration_seconds,
    )
    trace_id = mlflow.log_trace(
        name=name,
        request=inputs,
        response=outputs,
        attributes={
            "model": model,
            "duration_seconds": duration_seconds,
            "metadata": metadata,
        },
        tags={key: _stringify_tag_value(value) for key, value in metadata.items()},
        execution_time_ms=int(duration_seconds * 1000),
    )
    logging.info("MLflow trace logged name=%s trace_id=%s", name, trace_id)
