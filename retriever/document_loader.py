from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document
from pptx import Presentation
from io import BytesIO

def load_and_split_pdf(pdf_bytes):
    """Load and split PDF file."""
    loader = PyPDFLoader(BytesIO(pdf_bytes.read()))
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(docs)

def load_and_split_docx(docx_bytes):
    """Load and split DOCX file."""
    doc = Document(BytesIO(docx_bytes.read()))
    text = "\n".join([para.text for para in doc.paragraphs])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_text(text)

def load_and_split_pptx(pptx_bytes):
    """Load and split PPTX file."""
    prs = Presentation(BytesIO(pptx_bytes.read()))
    text = "\n".join([slide.shapes.text for slide in prs.slides if hasattr(slide.shapes, "text")])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_text(text)
