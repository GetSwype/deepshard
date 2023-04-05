import gradio as gr
import transformers
import torch
import os

    
def generate_text(
    model,
    text,
    temperature, 
    top_p, 
    top_k, 
    repetition_penalty, 
    max_new_tokens,
    progress=gr.Progress(track_tqdm=True)
):
    assert model is not None
    tokenizer = "swype/deepshard-13B-raw"
    print(model)

    model = transformers.LlamaForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
    )
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        tokenizer,
    )

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    generation_config = transformers.GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        num_beams=1,
    )

    with torch.no_grad():
        output = model.generate(  # type: ignore
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            generation_config=generation_config
        )[0].cuda()

    return tokenizer.decode(output, skip_special_tokens=True).strip()


def inference_tab():
    with gr.Tab("Inference"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                        model = gr.Dropdown(
                            label="Models",
                        )
                        refresh_models_list = gr.Button(
                            "Reload Models",
                            elem_id="refresh-button"
                        )
                inference_text = gr.Textbox(lines=7, label="Input Text")   
            inference_output = gr.Textbox(lines=12, label="Output Text")
        with gr.Row():
            with gr.Column():
                #  temperature, top_p, top_k, repeat_penalty, max_new_tokens
                temperature = gr.Slider(
                    minimum=0.01, maximum=1.99, value=0.4, step=0.01,
                    label="Temperature",
                    info="Controls the 'temperature' of the softmax distribution during sampling. Higher values (e.g., 1.0) make the model generate more diverse and random outputs, while lower values (e.g., 0.1) make it more deterministic and focused on the highest probability tokens."
                )

                top_p = gr.Slider(
                    minimum=0, maximum=1, value=0.3, step=0.01,
                    label="Top P",
                    info="Sets the nucleus sampling threshold. In nucleus sampling, only the tokens whose cumulative probability exceeds 'top_p' are considered  for sampling. This technique helps to reduce the number of low probability tokens considered during sampling, which can lead to more diverse and coherent outputs."
                )

                top_k = gr.Slider(
                    minimum=0, maximum=200, value=50, step=1,
                    label="Top K",
                    info="Sets the number of top tokens to consider during sampling. In top-k sampling, only the 'top_k' tokens with the highest probabilities are considered for sampling. This method can lead to more focused and coherent outputs by reducing the impact of low probability tokens."
                )

                repeat_penalty = gr.Slider(
                    minimum=0, maximum=2.5, value=1.0, step=0.01,
                    label="Repeat Penalty",
                    info="Applies a penalty to the probability of tokens that have already been generated, discouraging the model from repeating the same words or phrases. The penalty is applied by dividing the token probability by a factor based on the number of times the token has appeared in the generated text."
                )

                max_new_tokens = gr.Slider(
                    minimum=0, maximum=4096, value=50, step=1,
                    label="Max New Tokens",
                    info="Limits the maximum number of tokens generated in a single iteration."
                )
            with gr.Column():
                with gr.Row():
                    generate_btn = gr.Button(
                        "Generate", variant="primary", label="Generate", 
                    )
            
        generate_btn.click(
            fn=generate_text,
            inputs=[
                model,
                inference_text,
                temperature,
                top_p,
                top_k,
                repeat_penalty,
                max_new_tokens
            ],
            outputs=inference_output,
        )

        def update_models_list():
            return gr.Dropdown.update(choices=["None"] + [
                d for d in os.listdir() if os.path.isdir(d) and d.startswith('model-')
            ] + ["swype/deepshard-13B-raw"], value="None")
            
        refresh_models_list.click(
            update_models_list,  
            inputs=None, 
            outputs=model,
        )