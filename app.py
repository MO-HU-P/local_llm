import gradio as gr
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch
import os
from dotenv import load_dotenv


load_dotenv()

access_token = os.getenv("CLI_TOKEN")

model_name_or_path = "mmnga/ELYZA-japanese-Llama-2-7b-fast-instruct-AWQ-calib-ja-100k"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoAWQForCausalLM.from_quantized(model_name_or_path, safetensors=True, fuse_layers=True)


def generate_response(user_inputs, history, system_prompt, max_tokens, temperature, top_p, top_k, repetition_penalty):

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = system_prompt
    text = user_inputs

    prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
        bos_token=tokenizer.bos_token,
        b_inst=B_INST,
        system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
        prompt=text,
        e_inst=E_INST,
    )

    with torch.no_grad():
        token_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()

        output_ids = model.generate(
            token_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
        )

    output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)
    
    return output

iface = gr.ChatInterface(fn=generate_response, 
                     additional_inputs=[
                         gr.Textbox("あなたは優秀な日本語アシスタントです。", label="system_prompt"), 
                         gr.Slider(50, 500, 100, label="max_tokens"),
                         gr.Slider(0, 1.0, 0.7, label="temperature"),
                         gr.Slider(0.5, 0.95, 0.95, label="top_p"),
                         gr.Slider(1.0, 99, 40, label="top_k"),
                         gr.Slider(1.0, 1.5, 1.1, label="repetition_penalty")
                     ], 
                     title="Chat with ELYZA-japanese-Llama-2-7b-instruct",
                     description="ELYZAに何でも質問してみてください。",
                     theme="soft",
                     retry_btn=None,
                     undo_btn="Delete Previous",
                     clear_btn="Clear",
                    )

iface.launch()

