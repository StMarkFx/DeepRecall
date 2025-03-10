from retriever.vector_store import load_vector_store
from models.deepseek_model import deepseek_chat

# Load FAISS retriever
retriever = load_vector_store()

def chat_with_bot(query):
    """Retrieves relevant documents and generates a DeepSeek response."""
    retrieved_docs = []
    
    if retriever:
        retrieved_docs = retriever.get_relevant_documents(query)

    # Extract text from retrieved documents
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Append document context to user query
    enhanced_query = f"Context from documents:\n{context}\n\nUser Query: {query}"

    # Generate response from DeepSeek
    response = deepseek_chat(enhanced_query)
    
    return response, retrieved_docs  # Return chatbot response + retrieved docs
