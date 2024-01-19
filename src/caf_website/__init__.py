import gradio as gr

from . import callbacks as cb
from . import strings
from .models.catalog import Classifier, Embedder

with gr.Blocks(title=strings.TITLE) as demo:
    gr.Markdown(strings.INTRO)

    with gr.Row():
        with gr.Column(variant="panel"):
            model_selector = gr.Dropdown(
                label="Select a model",
                info="Choose between the different classifier models",
                choices=[model.value for model in Classifier],
            )

            embedding_selector = gr.Radio(
                label="Select an embedding",
                info="Choose between the different embeddings",
                choices=[emb.value for emb in Embedder],
            )

        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Maintenance issue",
                info="Enter a maintenance issue to classify.",
                placeholder="Enter some text...",
                lines=4,
            )

            submit_button = gr.Button(
                value="Submit",
                size="lg",
                variant="primary",
            )

    with gr.Row():
        labels = gr.Label(
            visible=False,
        )

    # Event handlers
    submit_button.click(
        cb.show_labels,
        inputs=[model_selector, embedding_selector, text_input],
        outputs=[labels],
    )

    submit_button.click(
        cb.predict,
        inputs=[model_selector, embedding_selector, text_input],
        outputs=[labels],
    )
