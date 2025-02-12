import streamlit as st
import os
from retriever.vector_store import process_and_store_documents, get_retriever
from models.deepseek_wrapper import DeepSeekChat

st.title("📚 DeepSeek RAG Chatbot")
st.write("Upload PDFs, and chat with the extracted knowledge!")

chatbot = DeepSeekChat()

# Upload multiple PDFs
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    file_paths = []
    os.makedirs("data/uploaded_files", exist_ok=True)
    
    for file in uploaded_files:
        file_path = os.path.join("data/uploaded_files", file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        file_paths.append(file_path)
    
    process_and_store_documents(file_paths)
    st.success(f"Processed {len(uploaded_files)} document(s).")

retriever = get_retriever()

# Chat interface
query = st.text_input("Ask a question:")

if query:
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])
    
    response = chatbot.chat(f"Context:\n{context}\n\nQuestion: {query}")
    
    st.write("**DeepSeek:**", response)

# Clear chat memory
if st.button("Clear Chat Memory"):
    chatbot.clear_history()
    st.success("Chat memory cleared.")
