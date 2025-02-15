import sys
import os
import streamlit as st
sys.path.append(os.path.abspath("retriever"))

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
    st.sidebar.write("📄 Processing documents... (this may take a few seconds)")
    st.session_state.retriever = process_documents(uploaded_files)
    st.sidebar.success("✅ Documents indexed!")

st.title("🤖 DeepRecall - Chat with Your Files")

# Function to generate AI response
def generate_response(prompt):
    """Generate AI response while ensuring English output."""
    history = st.session_state.chat_history[-5:]  # Keep last 5 messages only
    response = deepseek_chat(prompt, history)
    
    # Ensure response is in English
    if not response.isascii():
        response = "Sorry, I can only respond in English. Let me try again..."
    
    return response

# Function to retrieve document-related context
def retrieve_context(query):
    """Retrieve relevant document context if available."""
    retriever = st.session_state.retriever
    if retriever:
        docs = retriever.similarity_search(query, k=3)  # Get top 3 relevant docs
        context = "\n\n".join([doc.page_content for doc in docs])
        return context if context else ""
    return ""

# Chat Interface
user_input = st.chat_input("Ask me something...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)  # Show user message instantly

    with st.spinner("Thinking..."):  
        document_context = retrieve_context(user_input)
        full_prompt = f"Context: {document_context}\n\nUser: {user_input}" if document_context else user_input
        response = generate_response(full_prompt)

    with st.chat_message("ai"):
        st.markdown(response)

    # Update chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "ai", "content": response})

# Preserve chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
