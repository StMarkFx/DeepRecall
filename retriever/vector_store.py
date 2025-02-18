from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
import os
import streamlit as st
from PyPDF2 import PdfReader
from pptx import Presentation
from docx import Document as DocxDocument  # DOCX support
from concurrent.futures import ThreadPoolExecutor
from pdfminer.high_level import extract_text

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
    """Efficiently extract text from PDFs, PPTX, and DOCX."""
    text = ""

    if file.name.endswith(".pdf"):
        text = extract_text(file)

    elif file.name.endswith(".pptx"):
        ppt = Presentation(file)
        extracted_text = []
        for slide in ppt.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.has_text_frame:
                    extracted_text.append(shape.text)
        text = "\n".join(extracted_text)

    elif file.name.endswith(".docx"):
        doc = DocxDocument(file)
        extracted_text = [para.text for para in doc.paragraphs if para.text.strip()]
        text = "\n".join(extracted_text)

    return text

def process_documents(uploaded_files):
    """Process files in parallel and add to FAISS vector store."""
    if not uploaded_files:
        return None

    docs = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(extract_text_from_file, uploaded_files)

    for file, text in zip(uploaded_files, results):
        if text:
            docs.append(Document(page_content=text, metadata={"source": file.name}))

    if not docs:
        return None

    embedding = OllamaEmbeddings(model="deepseek-r1:1.5b")

    # Load existing FAISS index if available
    if os.path.exists(VECTOR_DB_PATH):
        existing_faiss = FAISS.load_local(VECTOR_DB_PATH, embedding)
        existing_faiss.add_documents(docs)
        existing_faiss.save_local(VECTOR_DB_PATH)
        return existing_faiss  # Return updated retriever
    else:
        new_faiss = FAISS.from_documents(docs, embedding)
        new_faiss.save_local(VECTOR_DB_PATH)
        return new_faiss  # Return new retriever

