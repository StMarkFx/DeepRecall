from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os
import config

def process_and_store_documents(pdf_files):
    """Loads multiple PDFs, extracts text, splits, and stores embeddings dynamically."""
    docs = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    texts = text_splitter.split_documents(docs)

    # Create or update vector store
    embedding = OllamaEmbeddings(model=config.MODEL_NAME)
    vectorstore = Chroma.from_documents(texts, embedding, persist_directory=config.VECTOR_DB_PATH)
    vectorstore.persist()

def get_retriever():
    """Loads the stored embeddings from ChromaDB."""
    embedding = OllamaEmbeddings(model=config.MODEL_NAME)
    return Chroma(persist_directory=config.VECTOR_DB_PATH, embedding_function=embedding).as_retriever()
