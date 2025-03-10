from retriever.vector_store import VectorStore
from models.deepseek_model import DeepSeekModel

class Chat:
    def __init__(self):
        self.retriever = VectorStore()
        self.model = DeepSeekModel()

    def get_response(self, query):
        relevant_indices = self.retriever.search(query)
        context = " ".join([f"Doc {i+1}" for i in relevant_indices])  # Placeholder for actual text retrieval
        return self.model.generate_response(query, context)
