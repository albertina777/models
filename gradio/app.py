import gradio as gr
import requests
import os
from langchain.vectorstores.pgvector import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

# Constants
API_URL = "https://gpt-ds-model.apps.cluster-v82jh.v82jh.sandbox1208.opentlc.com/v1/completions"
MODEL_NAME = "gpt"

# PGVector setup with psycopg3
DB_CONNECTION_STRING = "postgresql+psycopg://vectordb:vectordb@postgresql-service.pgvector.svc.cluster.local:5432/vectordb"
DB_COLLECTION_NAME = "documents_test"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Load embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vector_store = PGVector.from_existing_index(
    embedding=embeddings,
    collection_name=DB_COLLECTION_NAME,
    connection_string=DB_CONNECTION_STRING,
    use_jsonb=True
)

# Prompt Template
def construct_prompt(context, question):
    return f"""You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:"""

# RAG + LLM Call
def generate_response(question):
    # Step 1: Retrieve relevant documents
    docs = vector_store.similarity_search(question, k=4)
    context = "\n".join([doc.page_content for doc in docs])

    # Step 2: Construct prompt
    full_prompt = construct_prompt(context=context, question=question)

    # Step 3: Call LLM API
    payload = {
        "prompt": full_prompt,
        "model": MODEL_NAME,
        "temperature": 0.7,
        "max_tokens": 200
    }
    headers = {"Connection": "keep-alive"}
    response = requests.post(API_URL, json=payload, headers=headers)
    return response.json().get("choices", [{}])[0].get("text", "No response")

# Gradio UI
demo = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Enter your question"),
    outputs=gr.Textbox(label="Response"),
    title="Gradio with RAG and GPT2"
)

demo.launch()
