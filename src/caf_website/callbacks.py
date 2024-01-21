from typing import Optional

import gradio as gr

from . import pipeline as pl
from .models.catalog import (Classifier, Embedder, parse_classifier,
                             parse_embedder)
from .strings import DBSCAN_CLUSTER_LABELS


def predict(
    model_selector: list | str, embedding_selector: Optional[str], text_input: str
) -> None | dict[str, float]:
    """Predict the labels for the input text using the selected model and embedding."""
    if not _validate_input(model_selector, embedding_selector, text_input):
        return None

    assert isinstance(model_selector, str)
    assert isinstance(embedding_selector, str)

    embedding: Embedder = parse_embedder(embedding_selector)
    model: Classifier = parse_classifier(model_selector)

    rawpreds = pl.predict(text_input, embedding, model)

    return {
        DBSCAN_CLUSTER_LABELS[k]: v for k, v in enumerate(rawpreds.values()) if v > 0.3
    }


def show_labels(
    model_selector: list | str, embedding_selector: Optional[str], text_input: str
):
    """Show the labels component if the input is valid."""
    if _validate_input(model_selector, embedding_selector, text_input):
        return gr.update(visible=True)
    else:
        return None


def _validate_input(
    model_selector: list | str, embedding_selector: Optional[str], text_input: str
) -> bool:
    """Check if the input from the components is valid."""
    if model_selector is None or type(model_selector) == list:
        return False
    elif embedding_selector is None:
        return False
    elif not any(text_input):
        return False
    else:
        return True
