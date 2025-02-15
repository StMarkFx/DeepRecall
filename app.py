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

# User Input
user_input = st.chat_input("Ask me anything...")

if user_input:
    retriever = st.session_state.retriever
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

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "pptx"])
if uploaded_file:
    process_documents(uploaded_file)  # Ensure this does not reset session state
    st.success("File uploaded and processed!")

# Preserve chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# When user sends a message
user_input = st.chat_input("Ask me something...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        response = generate_response(user_input)

    st.session_state.messages.append({"role": "ai", "content": response})
    with st.chat_message("ai"):
        st.markdown(response)
