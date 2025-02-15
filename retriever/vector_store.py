from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
import os
import streamlit as st
from PyPDF2 import PdfReader
from pptx import Presentation

VECTOR_DB_PATH = "data/faiss_index"

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded files (txt, pdf, pptx)."""
    file_extension = uploaded_file.name.split(".")[-1]

    if file_extension == "txt":
        return uploaded_file.read().decode("utf-8")  # Read as text

    elif file_extension == "pdf":
        pdf_reader = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
        return text.strip()

    elif file_extension == "pptx":
        ppt = Presentation(uploaded_file)
        text = "\n".join([slide.shapes.text for slide in ppt.slides if hasattr(slide, "shapes")])
        return text.strip()

    else:
        st.warning(f"Unsupported file format: {file_extension}")
        return None

def process_documents(uploaded_files):
    """Process and convert uploaded files into LangChain Documents before adding them to FAISS."""
    if not uploaded_files:
        return None

    docs = []
    for file in uploaded_files:
        text = extract_text_from_file(file)
        if text:
            docs.append(Document(page_content=text, metadata={"source": file.name}))

    if not docs:
        return None

    embedding = OllamaEmbeddings(model="deepseek-r1:1.5b")
    return FAISS.from_documents(docs, embedding)
