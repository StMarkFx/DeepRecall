import re
import sys
import os
import streamlit as st
sys.path.append(os.path.abspath("retriever"))

from models.deepseek_model import deepseek_chat
from retriever.vector_store import load_vector_store, process_documents
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever  # Import retriever

st.set_page_config(page_title="DeepRecall", layout="wide")


# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = load_vector_store()  # ✅ Uses the fixed `load_vector_store()` method


# Sidebar for File Upload
st.sidebar.title("📂 Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs, DOCX, or PPTX", 
                                          type=["pdf", "docx", "pptx"], 
                                          accept_multiple_files=True)

if uploaded_files:
    st.sidebar.write("📄 Processing documents... (this may take a few seconds)")
    
    try:
        vector_store = process_documents(uploaded_files)
        if vector_store:
            st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        st.sidebar.success("✅ Documents indexed!")
    except Exception as e:
        st.sidebar.error(f"Error processing documents: {str(e)}")


st.title("🤖 DeepRecall - Chat with Your Files")

# Function to generate AI response
def generate_response(prompt):
    """Generate AI response while ensuring English output and better accuracy."""
    history = st.session_state.chat_history[-5:]  # Keep last 5 messages only
    context = retrieve_context(prompt)

    fixed_prompt = f"""
    You are an AI assistant with access to the following documents:

    {context}

    When answering, prioritize using information from the retrieved documents above. 
    If no relevant documents exist, say "I could not find relevant information." 
    Be concise, clear, and avoid speculation.

    User: {prompt}

    Respond concisely in clear English. If defining a term, provide a dictionary-style explanation.
    """

    response = deepseek_chat(fixed_prompt, history)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return response



# Function to retrieve document-related context
def retrieve_context(query):
    retriever = st.session_state.retriever
    if retriever:
        docs = retriever.get_relevant_documents(query)
        if docs:
            formatted_context = "\n\n".join([f"📄 {doc.metadata['source']}:\n{doc.page_content[:500]}" for doc in docs])
            return formatted_context
    return ""



# 🔥 Display existing chat history first**
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 🚀\ Process new user input**
user_input = st.chat_input("Ask me something...")
if user_input:
    # Show user message instantly
    with st.chat_message("user"):
        st.markdown(user_input)

    # Store user message in chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Generate response
    with st.spinner("Thinking..."):
        response = generate_response(user_input)  

    # Store AI response in chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Show AI response
    with st.chat_message("assistant"):
        st.markdown(response)
