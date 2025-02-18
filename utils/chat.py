import streamlit as st
from models.deepseek_model import deepseek_chat
from app import retrieve_context


def generate_response(user_query):
    """Retrieve relevant documents & generate a response."""
    retriever = st.session_state.get("retriever", None)
    
    context = retrieve_context(user_query) if retriever else ""

    chat_history = st.session_state.get("chat_history", [])
    history = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]

    # Include retrieved context
    prompt = f"Context: {context}\n\nUser: {user_query}\nAI:"

    response = deepseek_chat(prompt, history=history)

    return response
