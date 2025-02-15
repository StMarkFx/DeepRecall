import streamlit as st
from models.deepseek_model import deepseek_chat


def generate_response(user_query):
    """Retrieve relevant documents & generate a response."""
    retriever = st.session_state.retriever  

    if retriever:
        relevant_docs = retriever.get_relevant_documents(user_query)
        context = "\n".join([doc.page_content for doc in relevant_docs])
    else:
        context = ""

    prompt = f"Context: {context}\n\nUser: {user_query}\nAI:"
    response = deepseek_chat(prompt)  # Calls the AI model
    return response
