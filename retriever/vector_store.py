from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import pickle
import os

VECTOR_DB_PATH = "data/faiss_index"

# Load FAISS
def load_vector_store():
    if os.path.exists(VECTOR_DB_PATH):
        with open(VECTOR_DB_PATH, "rb") as f:
            return pickle.load(f)
    return None

# Save FAISS
def save_vector_store(vectorstore):
    with open(VECTOR_DB_PATH, "wb") as f:
        pickle.dump(vectorstore, f)

def get_retriever(docs):
    """Create FAISS vector store retriever."""
    embedding = OllamaEmbeddings(model="deepseek-r1:1.5b")
    vectorstore = FAISS.from_documents(docs, embedding)
    save_vector_store(vectorstore)
    return vectorstore.as_retriever()
