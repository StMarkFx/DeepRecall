from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
import os
import streamlit as st
from PyPDF2 import PdfReader
from pptx import Presentation
from docx import Document as DocxDocument  # Import DOCX processing

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
    """Extract text from PDFs, DOCX, and PPTX."""
    text = ""

    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

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
        text = "\n".join([para.text for para in doc.paragraphs])

    return text


def process_documents(uploaded_files):
    """Process uploaded files and update FAISS vector store."""
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
    vectorstore = FAISS.from_documents(docs, embedding)

    # Save to FAISS for retrieval
    if not os.path.exists("data"):
        os.makedirs("data")
    vectorstore.save_local(VECTOR_DB_PATH)

    return vectorstore
