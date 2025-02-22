import streamlit as st
from models.deepseek_model import deepseek_chat
from app import retrieve_context

def generate_response(user_query):
    """Retrieve document context & generate AI response."""
    retriever = st.session_state.get("retriever", None)
    context = retrieve_context(user_query) if retriever else ""

    chat_history = st.session_state.get("chat_history", [])
    history = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]

    prompt = f"""
    You are answering user questions using document knowledge.
    If relevant information is found in the document, use it to answer.
    Otherwise, state that you do not have relevant information.

    Document Context:
    {context}

    User: {user_query}
    AI:
    """

    response = deepseek_chat(prompt, history=history)
    return response

