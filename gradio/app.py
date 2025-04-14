import os
from collections.abc import Generator
from queue import Empty, Queue
from threading import Thread
from typing import Optional, List

import gradio as gr
import requests
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.vectorstores.pgvector import PGVector

load_dotenv()

# Parameters
APP_TITLE = os.getenv('APP_TITLE', 'Talk with GPT2')
INFERENCE_SERVER_URL = os.getenv('INFERENCE_SERVER_URL')
MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', 15))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.01))
DB_CONNECTION_STRING = os.getenv('DB_CONNECTION_STRING')
DB_COLLECTION_NAME = os.getenv('DB_COLLECTION_NAME')

# Streaming callback
class QueueCallback(BaseCallbackHandler):
    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: any) -> None:
        return self.q.empty()

# Custom OpenAI-compatible LLM
class OpenAICompatibleLLM(LLM):
    inference_server_url: str
    model: str = "gpt"
    max_tokens: int = 15
    temperature: float = 0.7

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Truncate prompt if it's too long
        max_prompt_tokens = 1024 - self.max_tokens
        prompt_words = prompt.split()
        if len(prompt_words) > max_prompt_tokens:
            prompt = " ".join(prompt_words[:max_prompt_tokens])

        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            self.inference_server_url,
            headers=headers,
            json=payload
        )
        if response.status_code == 200:
            return response.json().get("choices", [{}])[0].get("text", "").strip()
        else:
            raise Exception(f"Request failed [{response.status_code}]: {response.text}")

    @property
    def _llm_type(self) -> str:
        return "openai-compatible"

# Utility to remove duplicate sources
def remove_source_duplicates(input_list):
    unique_list = []
    for item in input_list:
        if item.metadata['source'] not in unique_list:
            unique_list.append(item.metadata['source'])
    return unique_list

# Streaming generator
def stream(input_text) -> Generator:
    job_done = object()

    def task():
        resp = qa_chain({"query": input_text})
        sources = remove_source_duplicates(resp['source_documents'])
        if len(sources) != 0:
            q.put("\n*Sources:* \n")
            for source in sources:
                q.put("* " + str(source) + "\n")
        q.put(job_done)

    t = Thread(target=task)
    t.start()

    content = ""
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            if isinstance(next_token, str):
                content += next_token
                yield next_token, content
        except Empty:
            continue

# Queue for streaming
q = Queue()

# PGVector-based retrieval
embeddings = HuggingFaceEmbeddings()
store = PGVector(
    connection_string=DB_CONNECTION_STRING,
    collection_name=DB_COLLECTION_NAME,
    embedding_function=embeddings
)

# Load custom LLM
llm = OpenAICompatibleLLM(
    inference_server_url=INFERENCE_SERVER_URL,
    model="gpt",
    max_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE
)

# Prompt template
template = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant named HatBot answering questions about OpenShift AI, aka RHOAI.
You will be given a question you need to answer, and a context to provide you with information. You must answer the question based as much as possible on this context.
Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

Context: 
{context}

Question: {question} [/INST]
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# RAG setup
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 4, "score_threshold": 0.2}
    ),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True
)

# Gradio interface
def ask_llm(message, history):
    for next_token, content in stream(message):
        yield(content)

with gr.Blocks(title="RHOAI HatBot", css="footer {visibility: hidden}") as demo:
    chatbot = gr.Chatbot(
        show_label=False,
        avatar_images=(None, 'assets/robot-head.svg'),
        render=False
    )
    gr.ChatInterface(
        ask_llm,
        chatbot=chatbot,
        clear_btn=None,
        retry_btn=None,
        undo_btn=None,
        stop_btn=None,
        description=APP_TITLE
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name='0.0.0.0',
        share=False,
        favicon_path='./assets/robot-head.ico'
    )
