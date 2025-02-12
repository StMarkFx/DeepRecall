from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import config

def load_and_store_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    texts = text_splitter.split_documents(docs)
    
    embedding = OllamaEmbeddings(model=config.MODEL_NAME)
    db = FAISS.from_documents(texts, embedding)
    db.save_local(config.VECTOR_DB_PATH)

def get_retriever():
    embedding = OllamaEmbeddings(model=config.MODEL_NAME)
    return FAISS.load_local(config.VECTOR_DB_PATH, embedding).as_retriever()
