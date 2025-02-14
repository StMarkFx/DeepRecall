from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import os

VECTOR_DB_PATH = "data/faiss_index"

def process_documents(docs):
    """Process documents before adding them to FAISS."""
    if not docs:
        return None
    embedding = OllamaEmbeddings(model="deepseek-r1:1.5b")
    return FAISS.from_documents(docs, embedding)

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

def save_vector_store(vectorstore):
    """Save FAISS vector store."""
    if not os.path.exists("data"):
        os.makedirs("data")  
    vectorstore.save_local(VECTOR_DB_PATH)

def get_retriever(docs):
    """Create or load FAISS vector store retriever."""
    vectorstore = load_vector_store()
    
    if vectorstore is None and docs:
        vectorstore = process_documents(docs)
        save_vector_store(vectorstore)

    return vectorstore.as_retriever() if vectorstore else None
