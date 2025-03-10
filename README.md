# DeepRecall - Chat with Your Documents

DeepRecall is a **Retrieval-Augmented Generation (RAG)** chatbot that allows users to upload PDFs, DOCX, and PPTX files and query them using AI. The chatbot retrieves relevant document content using **FAISS** vector storage and generates responses using the **DeepSeek API**.

## 🚀 Features

- 📂 **Upload Documents**: Supports PDF, DOCX, and PPTX formats.
- 🔍 **Intelligent Search**: Retrieves the most relevant content using FAISS-based semantic search.
- 🤖 **AI-Powered Responses**: Uses DeepSeek to provide contextual and accurate answers.
- 💾 **Persistent Vector Store**: Indexed documents are stored and retrieved efficiently.
- 🏎 **Fast Processing**: Parallel document processing with multi-threading.

---

## 🏗️ Project Structure

```bash
DeepRecall/
│── retriever/                  # Core retrieval logic
│   ├── vector_store.py         # FAISS vector storage and retrieval
│   ├── document_loader.py      # Document parsing and text extraction
│── models/
│   ├── deepseek_model.py       # DeepSeek API call for chat responses
│── utils/
│   ├── chat.py 
│── data/                        # Stored FAISS vector index
│── app.py                       # Streamlit UI and chatbot logic
│── requirements.txt             # Dependencies
│── README.md                    # Project documentation
```

📦 Installation
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/DeepRecall.git
cd DeepRecall
2. Create a Virtual Environment (Optional but Recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
3. Install Dependencies
pip install -r requirements.txt
🛠️ Usage
1. Start the Application
bash
Copy
Edit
streamlit run app.py
2. Upload Documents
Click on "📂 Upload Documents" in the sidebar.
Select PDFs, DOCX, or PPTX files.
Wait for them to be indexed.
3. Ask Questions
Type your query in the chat input field.
The AI retrieves relevant context from the uploaded documents.
AI responds based on the retrieved content.

🧩 How It Works
Document Processing

PDFs, DOCX, and PPTX files are parsed.
Text is split into smaller chunks for efficient retrieval.
Vector Storage

Text chunks are embedded using DeepSeek-R1 embeddings.
FAISS stores these embeddings for fast semantic search.
Retrieval & Augmented Generation

When a user asks a question, the most relevant document chunks are retrieved.
The retrieved text is passed as context to the DeepSeek API.
DeepSeek generates a response that incorporates the document content.

🛠️ Key Components
vector_store.py
Loads and manages FAISS vector storage.
Extracts text from documents.
Generates and stores embeddings.
document_loader.py
Loads and splits documents into manageable chunks.
Uses pdfminer, python-pptx, and python-docx for parsing.
app.py
Streamlit-based UI for chat interactions.
Handles document uploads and query processing.
deepseek_model.py
Connects to the DeepSeek API for AI-generated responses.