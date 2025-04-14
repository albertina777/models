import os
import gradio as gr
import requests
import tiktoken
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.vectorstores.pgvector import PGVector
from typing import Optional, List

load_dotenv()

# Parameters
APP_TITLE = os.getenv('APP_TITLE', 'Talk with GPT2')
INFERENCE_SERVER_URL = os.getenv('INFERENCE_SERVER_URL')
MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', 15))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.01))
DB_CONNECTION_STRING = os.getenv('DB_CONNECTION_STRING')
DB_COLLECTION_NAME = os.getenv('DB_COLLECTION_NAME')

# Custom LLM class to match curl-style /v1/completions
class OpenAICompatibleLLM(LLM):
    inference_server_url: str
    model: str = "gpt"
    max_tokens: int = 15
    temperature: float = 0.7

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        tokenizer = tiktoken.get_encoding("gpt2")
        prompt_tokens = tokenizer.encode(prompt)
        max_prompt_tokens = 1024 - self.max_tokens
        if len(prompt_tokens) > max_prompt_tokens:
            prompt_tokens = prompt_tokens[:max_prompt_tokens]
            prompt = tokenizer.decode(prompt_tokens)

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
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("choices", [{}])[0].get("text", "").strip()
        else:
            raise Exception(f"Request failed [{response.status_code}]: {response.text}")

    @property
    def _llm_type(self) -> str:
        return "openai-compatible"

# Helper to remove duplicate sources
def remove_source_duplicates(input_list):
    seen = set()
    unique = []
    for item in input_list:
        src = item.metadata['source']
        if src not in seen:
            seen.add(src)
            unique.append(src)
    return unique

# Document store
embeddings = HuggingFaceEmbeddings()
store = PGVector(
    connection_string=DB_CONNECTION_STRING,
    collection_name=DB_COLLECTION_NAME,
    embedding_function=embeddings)

# LLM
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

# Chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 4, "score_threshold": 0.2 }),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True
)

# Gradio interface handler
def ask_llm(message, history):
    try:
        resp = qa_chain({"query": message})
        answer = resp["result"]
        sources = remove_source_duplicates(resp["source_documents"])

        if sources:
            source_text = "\n\n*Sources:*\n" + "\n".join([f"- {src}" for src in sources])
        else:
            source_text = ""

        return answer + source_text
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio app
with gr.Blocks(title="RHOAI HatBot", css="footer {visibility: hidden}") as demo:
    chatbot = gr.Chatbot(
        show_label=False,
        avatar_images=(None, 'assets/robot-head.svg'),
        render=False
    )
    gr.ChatInterface(
        fn=ask_llm,
        chatbot=chatbot,
        clear_btn=None,
        retry_btn=None,
        undo_btn=None,
        stop_btn=None,
        description=APP_TITLE
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False, favicon_path="assets/robot-head.ico")
