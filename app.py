import streamlit as st
from retriever.vector_store import process_documents
from utils.chat import chat_with_bot

# Streamlit App Title
st.title("📚 DeepRecall - AI Research Assistant")

# File Upload Section
st.sidebar.header("📂 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs, PPTX, or DOCX files",
    accept_multiple_files=True,
    type=["pdf", "pptx", "docx"]
)

# Process documents if uploaded
if uploaded_files:
    st.sidebar.write("🔄 Processing documents...")
    process_documents(uploaded_files)
    st.sidebar.success("✅ Documents indexed successfully!")

# Chat Interface
st.subheader("💬 Chat with DeepSeek AI")

# Conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box
query = st.chat_input("Ask a question about your documents...")
if query:
    # Show user query
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate AI response
    response, retrieved_docs = chat_with_bot(query)

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(response)

    # Store response in chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Show retrieved documents (optional)
    with st.expander("🔍 Retrieved Documents"):
        for doc in retrieved_docs:
            st.markdown(f"📄 **{doc.metadata['source']}**")
            st.write(doc.page_content[:500] + "...")  # Show preview
