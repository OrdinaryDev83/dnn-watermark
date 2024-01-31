import gradio as gr
import requests
import json
import base64
import os
import time
from service import detect_watermark, get_models_available

with gr.Blocks() as demo:
    gr.Markdown("# Watermark Detection")
    with gr.Row():
        model_dropdown = gr.Dropdown(choices=get_models_available(),value=get_models_available()[0], multiselect=False, label="Model to use", show_label=True)
    
    with gr.Row():
        image = gr.Image()
        image_output = gr.AnnotatedImage()

    deteck_btn = gr.Button("Detect Watermark")
    deteck_btn.click(detect_watermark, [model_dropdown, image], outputs=image_output)

    gr.Markdown("## Watermark Examples")
    with gr.Row():
        examples = gr.Examples([os.path.join(os.path.dirname(__file__), f"img_examples/000000000782.jpg")], inputs=image)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)