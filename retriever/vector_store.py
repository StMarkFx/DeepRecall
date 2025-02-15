from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
import os
import streamlit as st
from PyPDF2 import PdfReader
from pptx import Presentation

VECTOR_DB_PATH = "data/faiss_index"

def load_vector_store():
    """Load FAISS vector store if it exists."""
    if os.path.exists(VECTOR_DB_PATH):
        try:
            embedding = OllamaEmbeddings(model="deepseek-r1:1.5b")
            return FAISS.load_local(VECTOR_DB_PATH, embedding)
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return None
    return None


def extract_text_from_file(file):
    """Extract text from different file formats."""
    text = ""

    if file.name.endswith(".pptx"):
        ppt = Presentation(file)
        extracted_text = []

        for slide in ppt.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.has_text_frame:
                    extracted_text.append(shape.text)

        text = "\n".join(extracted_text)

    return text

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
