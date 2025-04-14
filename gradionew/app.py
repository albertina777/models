import os
import requests
import tiktoken
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.vectorstores.pgvector import PGVector
from langchain.callbacks.base import BaseCallbackHandler
from typing import Optional, List
import gradio as gr

load_dotenv()

# Parameters
APP_TITLE = os.getenv('APP_TITLE', 'Talk with GPT2')
INFERENCE_SERVER_URL = os.getenv('INFERENCE_SERVER_URL')
MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', 15))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.5))
DB_CONNECTION_STRING = os.getenv('DB_CONNECTION_STRING')
DB_COLLECTION_NAME = os.getenv('DB_COLLECTION_NAME')


# Custom LLM class to call OpenAI-compatible completions endpoint
class OpenAICompatibleLLM(LLM):
    inference_server_url: str
    model: str = "gpt"
    max_tokens: int = 15
    temperature: float = 0.5

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
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
                timeout=300
            )
            if response.status_code == 200:
                return response.json().get("choices", [{}])[0].get("text", "").strip()
            else:
                raise Exception(f"Request failed [{response.status_code}]: {response.text}")
        except requests.exceptions.Timeout:
            return "⚠️ The request timed out. Please try again."
        except Exception as e:
            return f"❌ Error: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "openai-compatible"


def remove_source_duplicates(input_list):
    seen_sources = set()
    unique_sources = []
    for item in input_list:
        source = item.metadata.get('source')
        if source and source not in seen_sources:
            seen_sources.add(source)
            unique_sources.append(source)
    return unique_sources


embeddings = HuggingFaceEmbeddings()
store = PGVector(
    connection_string=DB_CONNECTION_STRING,
    collection_name=DB_COLLECTION_NAME,
    embedding_function=embeddings
)

llm = OpenAICompatibleLLM(
    inference_server_url=INFERENCE_SERVER_URL,
    model="gpt",
    max_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE
)

template = """You are an assistant that answers questions based only on the context provided.

Context:
{context}

Q: {question}
A:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 4, "score_threshold": 0.2}
    ),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True
)


def ask_llm(message, history):
    response = qa_chain({"query": message})
    answer = response["result"]
    sources = remove_source_duplicates(response["source_documents"])
    if sources:
        answer += "\n\n**Sources:**\n" + "\n".join([f"- {src}" for src in sources])
    return answer


with gr.Blocks(title="RHOAI HatBot", css="footer {visibility: hidden}") as demo:
    chatbot = gr.Chatbot(
        show_label=False,
        avatar_images=(None, 'assets/robot-head.svg'),
        render=False
    )
    gr.ChatInterface(
        fn=ask_llm,
        chatbot=chatbot,
        description=APP_TITLE
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        share=False,
        favicon_path="./assets/robot-head.ico"
    )
