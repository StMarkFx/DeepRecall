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
│── data/                        # Stored FAISS vector index
│── app.py                       # Streamlit UI and chatbot logic
│── requirements.txt             # Dependencies
│── README.md                    # Project documentation

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