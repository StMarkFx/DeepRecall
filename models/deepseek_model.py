import ollama
from collections import deque

class DeepSeekModel:
    def __init__(self, model_name="deepseek", memory_size=5):
        self.model_name = model_name
        self.memory = deque(maxlen=memory_size)  # Store last 'n' messages

    def generate_response(self, query, context):
        # Save user query to memory
        self.memory.append({"role": "user", "content": query})
        
        # Construct conversation history
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.memory])
        
        # Fixed structured prompt
        prompt = f"""
        You are an AI research assistant helping students and researchers analyze documents.
        Follow these rules:
        - Answer queries based on retrieved documents and previous conversations.
        - If the query is unclear, ask clarifying questions.
        - Provide structured answers using bullet points, numbered lists, or paragraphs.
        - If relevant, cite sources from retrieved documents.
        - If asked, summarize documents or explain in simpler terms.
        
        Context from retrieved documents:
        {context}

        Conversation history:
        {history}

        User's new question:
        {query}

        Your response:
        """
        
        # Get response from DeepSeek
        response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        answer = response["message"]["content"]

        # Save AI response to memory
        self.memory.append({"role": "assistant", "content": answer})

        return answer
