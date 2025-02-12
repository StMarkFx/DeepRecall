import streamlit as st
import os
from retriever.vector_store import process_and_store_documents, get_retriever
from models.deepseek_model import DeepSeekChat

st.set_page_config(page_title="DeepSeek RAG Chatbot", layout="wide")

st.title("🤖 DeepSeek RAG Chatbot")
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

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
query = st.chat_input("Ask me anything about your uploaded documents...")

if query:
    # Display user message
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Retrieve relevant documents
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])

    # Generate bot response
    response = chatbot.chat(f"Context:\n{context}\n\nQuestion: {query}")

    # Display bot message
    st.session_state["messages"].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# Clear chat memory
if st.button("🗑 Clear Chat Memory"):
    chatbot.clear_history()
    st.session_state["messages"] = []
    st.success("Chat memory cleared.")
