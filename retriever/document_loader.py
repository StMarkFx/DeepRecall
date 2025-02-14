from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document
from pptx import Presentation
import tempfile

def load_and_split_pdf(pdf_bytes):
    """Load and split PDF file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_bytes.read())
        temp_file_path = temp_file.name
    
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(docs)

def load_and_split_docx(docx_bytes):
    """Load and split DOCX file."""
    doc = Document(docx_bytes)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_text(text)

def load_and_split_pptx(pptx_bytes):
    """Load and split PPTX file."""
    prs = Presentation(pptx_bytes)
    text = []

    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                text.append(shape.text)
    
    combined_text = "\n".join(text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_text(combined_text)
