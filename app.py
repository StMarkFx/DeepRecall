import sys
import os
sys.path.append(os.path.abspath("retriever"))

import streamlit as st
from models.deepseek_model import deepseek_chat
from retriever.vector_store import load_vector_store, process_documents

st.set_page_config(page_title="DeepRecall", layout="wide")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = load_vector_store()

# Sidebar for File Upload
st.sidebar.title("📂 Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs, DOCX, or PPTX", 
                                          type=["pdf", "docx", "pptx"], 
                                          accept_multiple_files=True)

if uploaded_files:
    st.sidebar.write("📄 Processing documents...")
    st.session_state.retriever = process_documents(uploaded_files)
    st.sidebar.success("✅ Documents indexed!")

st.title("🤖 DeepRecall - Chat with Your Files")

# Preserve chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Handling
user_input = st.chat_input("Ask me something...")
if user_input:
    # Show user input instantly
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        response = generate_response(user_input)  # AI response generation

    # Store AI response
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
