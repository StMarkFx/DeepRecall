from retriever.vector_store import load_vector_store
from models.deepseek_model import deepseek_chat
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import pickle
import os

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load FAISS retriever
retriever = load_vector_store()

# Initialize T5 model for summarization
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

def summarize_text(text):
    """Summarize text using T5 model."""
    input_ids = t5_tokenizer.encode("summarize: " + text, return_tensors="pt")
    output = t5_model.generate(input_ids, max_length=100)
    summary = t5_tokenizer.decode(output[0], skip_special_tokens=True)
    return summary

def cache_retrieved_docs(query, docs):
    """Cache retrieved documents for future queries."""
    cache_path = "cache/retrieved_docs.pkl"
    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    except FileNotFoundError:
        cache = {}

    cache[query] = docs
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

def load_cached_docs(query):
    """Load cached retrieved documents."""
    cache_path = "cache/retrieved_docs.pkl"
    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
            return cache.get(query, [])
    except FileNotFoundError:
        return []

def chat_with_bot(query):
    """Retrieves relevant documents and generates a DeepSeek response."""
    retrieved_docs = load_cached_docs(query)  # Check cache first

    if not retrieved_docs:
        try:
            if retriever:
                retrieved_docs = retriever.get_relevant_documents(query)
                logger.info(f"Retrieved {len(retrieved_docs)} documents.")
                cache_retrieved_docs(query, retrieved_docs)  # Cache retrieved docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return "Sorry, an error occurred while retrieving documents.", []

    # Extract and limit context from retrieved documents
    context = ""
    if retrieved_docs:
        # Summarize each document to limit context size
        summaries = [summarize_text(doc.page_content) for doc in retrieved_docs[:5]]
        context = "\n".join(summaries)
        logger.info("Context extracted from documents.")

    # Append document context to user query
    enhanced_query = f"Context from documents:\n{context}\n\nUser Query: {query}"

    try:
        # Generate response from DeepSeek
        response = deepseek_chat(enhanced_query)
        logger.info("Response generated successfully.")
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        response = "Sorry, an error occurred while generating a response."

    return response, retrieved_docs

# Example usage
if __name__ == "__main__":
    query = "What is the main topic of the documents?"
    response, docs = chat_with_bot(query)
    print(f"Response: {response}")
    print(f"Retrieved Docs: {len(docs)}")
