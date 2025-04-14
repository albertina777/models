#!/usr/bin/env python
# coding: utf-8

import os
import requests
from pathlib import Path

from langchain.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector

# -------------------------------
# Installation (if needed)
# -------------------------------
# !pip install langchain langchain-community sentence-transformers transformers torch pypdf requests beautifulsoup4 psycopg2-binary sqlalchemy

# -------------------------------
# Config
# -------------------------------
product_version = "2-latest"
CONNECTION_STRING = "postgresql+psycopg://vectordb:vectordb@postgresql-service.pgvector.svc.cluster.local:5432/vectordb"
COLLECTION_NAME = "documents_test"

# -------------------------------
# Prepare PDF URLs
# -------------------------------
documents = [
    "release_notes",
    "introduction_to_red_hat_openshift_ai",
    "getting_started_with_red_hat_openshift_ai_self-managed",   
]

pdfs = [
    f"https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/{product_version}/pdf/{doc}/red_hat_openshift_ai_self-managed-{product_version}-{doc}-en-us.pdf"
    for doc in documents
]

pdfs_to_urls = {
    f"red_hat_openshift_ai_self-managed-{product_version}-{doc}-en-us":
    f"https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/{product_version}/html-single/{doc}/index"
    for doc in documents
}

# -------------------------------
# Download PDFs
# -------------------------------
pdf_folder_path = f"./rhoai-doc-{product_version}"
os.makedirs(pdf_folder_path, exist_ok=True)

for pdf in pdfs:
    try:
        response = requests.get(pdf)
        if response.status_code != 200:
            print(f"[WARN] Skipped {pdf} due to status code {response.status_code}")
            continue
        with open(os.path.join(pdf_folder_path, pdf.split('/')[-1]), 'wb') as f:
            f.write(response.content)
    except Exception as e:
        print(f"[ERROR] Exception while downloading {pdf}: {e}")

# -------------------------------
# Load Documents
# -------------------------------
pdf_loader = PyPDFDirectoryLoader(pdf_folder_path)
pdf_docs = pdf_loader.load()

pdf_new_loader = PyPDFDirectoryLoader("./my_docs")
pdf_new_docs = pdf_new_loader.load()

website_loader = WebBaseLoader([
    "https://ai-on-openshift.io/getting-started/openshift/",
    "https://ai-on-openshift.io/getting-started/opendatahub/",
    "https://ai-on-openshift.io/getting-started/openshift-ai/",
    "https://ai-on-openshift.io/odh-rhoai/configuration/",
    "https://ai-on-openshift.io/odh-rhoai/custom-notebooks/",
    "https://ai-on-openshift.io/odh-rhoai/nvidia-gpus/",
    "https://ai-on-openshift.io/odh-rhoai/custom-runtime-triton/",
    "https://ai-on-openshift.io/odh-rhoai/openshift-group-management/",
    "https://ai-on-openshift.io/tools-and-applications/minio/minio/",
    "https://access.redhat.com/articles/7047935",
    "https://access.redhat.com/articles/rhoai-supported-configs",
])
website_docs = website_loader.load()

# -------------------------------
# Metadata Injection & Cleaning
# -------------------------------
def clean_documents(docs):
    for doc in docs:
        if doc.page_content:
            doc.page_content = doc.page_content.replace('\x00', '')
    return docs

for doc in pdf_docs:
    doc.metadata["source"] = pdfs_to_urls.get(Path(doc.metadata["source"]).stem, "unknown")

for doc in pdf_new_docs:
    name = Path(doc.metadata["source"]).stem
    doc.metadata["source"] = f"custom:{name}"

pdf_docs = clean_documents(pdf_docs)
pdf_new_docs = clean_documents(pdf_new_docs)
website_docs = clean_documents(website_docs)

# -------------------------------
# Split Documents
# -------------------------------
docs = pdf_docs + pdf_new_docs + website_docs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=40)
all_splits = text_splitter.split_documents(docs)

# Clean again after splitting
for doc in all_splits:
    doc.page_content = doc.page_content.replace('\x00', '')

# -------------------------------
# Index to Vector DB
# -------------------------------
embeddings = HuggingFaceEmbeddings()
db = PGVector.from_documents(
    documents=all_splits,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    pre_delete_collection=True
)

# -------------------------------
# Test a Query
# -------------------------------
query = "Is Oracle a Red Hat CCSP partner?"
docs_with_score = db.similarity_search_with_score(query)

# -------------------------------
# Display All Results
# -------------------------------
print("\n[INFO] All Results:\n")
for doc, score in docs_with_score:
    print("-" * 80)
    print(f"Score: {score:.4f}")
    print(f"Source: {doc.metadata.get('source', 'unknown')}")
    print(doc.page_content[:500])

# -------------------------------
# Display Custom PDF Results Only
# -------------------------------
print("\n[INFO] Filtering only results from custom documents:\n")
for doc, score in docs_with_score:
    if doc.metadata.get("source", "").startswith("custom:"):
        print("-" * 80)
        print(f"[CUSTOM PDF] Score: {score:.4f}")
        print(f"Source: {doc.metadata['source']}")
        print(doc.page_content[:500])
