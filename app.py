import streamlit as st
from models.deepseek_model import deepseek_chat
from retriever.vector_store import load_vector_store

st.set_page_config(page_title="DeepRecall", layout="wide")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🤖 DeepRecall - Chat with Your Files")

# User Input
user_input = st.chat_input("Ask me anything...")

# Load vector store (FAISS)
retriever = load_vector_store()

if user_input:
    # Retrieve relevant docs
    docs = retriever.invoke(user_input) if retriever else []
    
    # Format history
    chat_history = [{"role": "assistant", "content": msg} if i % 2 else {"role": "user", "content": msg}
                    for i, msg in enumerate(st.session_state.chat_history)]
    
    # Get response
    response = deepseek_chat(user_input, chat_history)
    
    # Update chat history
    st.session_state.chat_history.extend([user_input, response])
    
    # Display chat
    for i, msg in enumerate(st.session_state.chat_history):
        role = "user" if i % 2 == 0 else "assistant"
        st.chat_message(role).write(msg)
