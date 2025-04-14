import gradio as gr
import requests

API_URL = "https://vllm-llama3-3b-instruct-ds-model-test.apps.ocp.infer.com/v1/completions"
TOKEN ="xyeeje"

def generate_response(prompt):
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    response = requests.post(API_URL, json={"prompt": prompt, "model": "vllm-llama3-3b-instruct", "temperature": 0.7,"max_tokens": 256}, headers=headers, verify=False)
    return response.json()

demo = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Enter your prompt"),
    outputs=gr.Textbox(label="Response"),
    title="TinyLLM Chatbot",
)

demo.launch()
