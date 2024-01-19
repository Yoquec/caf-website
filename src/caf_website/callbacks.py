from typing import Optional

import gradio as gr


def predict(
    model_selector: list | str, embedding_selector: Optional[str], text_input: str
) -> None | dict[str, float]:
    if not _validate_input(model_selector, embedding_selector, text_input):
        return None

    return {
        "Arena la pieza": 0.7,
        "Anomalía; recomendable reportarla": 0.2,
        "Cargar software": 0.1,
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
