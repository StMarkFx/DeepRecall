import streamlit as st
from retriever.vector_store.py import get_retriever
from models.deepseek_model import deepseek_chat

st.title("DeepSeek RAG Chatbot")
st.write("Upload a PDF and ask questions!")

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with open("data/uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    st.success("File uploaded successfully! You can now ask questions.")

    retriever = get_retriever()
    query = st.text_input("Ask a question:")

    if query:
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])
        
        response = deepseek_chat(f"Context:\n{context}\n\nQuestion: {query}")
        st.write(response)
