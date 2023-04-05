from train import training_tab
# from inference import inference_tab
from dotenv import load_dotenv

import transformers
import gradio as gr
import argparse
import os
import torch

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

with gr.Blocks(
    css="#refresh-button { max-width: 32px }", 
    title="Simple LLaMA Finetuner") as demo:
        training_tab()
        # inference_tab()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple LLaMA Finetuner")
    parser.add_argument("-s", "--share", action="store_true", help="Enable sharing of the Gradio interface")
    args = parser.parse_args()

    demo.queue().launch(share=args.share)