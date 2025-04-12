import gradio as gr
import requests

API_URL = "https://gpt-ds-model.apps.cluster-v82jh.v82jh.sandbox1208.opentlc.com/v1/completions"
MODEL_NAME = "gpt"

def generate_response(prompt):
    response = requests.post(API_URL, json={"prompt": prompt, "model": MODEL_NAME, "temperature": 0.7,"max_tokens": 50})
    return response.json()

demo = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Enter your prompt"),
    outputs=gr.Textbox(label="Response"),
    title="Gradio with RAG and GPT2",
)

demo.launch()
