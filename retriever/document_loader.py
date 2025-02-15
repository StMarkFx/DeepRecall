from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from langchain_core.documents import Document
from docx import Document
from pptx import Presentation
import tempfile
import os

def load_and_split_pdf(pdf_bytes):
    """Load and split PDF file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_bytes.read())
        temp_file_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
    finally:
        os.remove(temp_file_path)  # Cleanup temp file
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(docs)

def load_and_split_docx(docx_bytes):
    """Load and split DOCX file."""
    doc = Document(docx_bytes)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    return [LangchainDocument(page_content=chunk) for chunk in chunks]


def load_and_split_pptx(pptx_bytes):
    """Load and split PPTX file into Document objects."""
    prs = Presentation(pptx_bytes)
    text = []

    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                text.append(shape.text)
    
    combined_text = "\n".join(text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_text(combined_text)

    # Wrap chunks in Document objects
    return [Document(page_content=chunk) for chunk in text_chunks]
