import os
import requests
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.vectorstores.pgvector import PGVector
from typing import Optional, List
import gradio as gr

load_dotenv()

# Parameters
APP_TITLE = os.getenv('APP_TITLE', 'Talk with GPT2')
INFERENCE_SERVER_URL = os.getenv('INFERENCE_SERVER_URL')
DB_CONNECTION_STRING = os.getenv('DB_CONNECTION_STRING')
DB_COLLECTION_NAME = os.getenv('DB_COLLECTION_NAME')
MODEL_PATH = "/models/all-mpnet-base-v2"

# Custom LLM class to call OpenAI-compatible completions endpoint
class OpenAICompatibleLLM(LLM):
    inference_server_url: str
    model: str = "gpt"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            # Manual prompt length control
            words = prompt.split()
            if len(words) > 500:
                prompt = " ".join(words[:500])

            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": 0.2,
                "max_tokens": 150,
                "top_p": 0.75,
                "do_sample": True,
                "repetition_penalty": 1.2,
                "num_return_sequences": 1
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

embeddings = HuggingFaceEmbeddings(model_name=MODEL_PATH)
store = PGVector(
    connection_string=DB_CONNECTION_STRING,
    collection_name=DB_COLLECTION_NAME,
    embedding_function=embeddings
)

llm = OpenAICompatibleLLM(
    inference_server_url=INFERENCE_SERVER_URL,
    model="gpt"
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
