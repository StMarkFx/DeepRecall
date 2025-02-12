import ollama

class DeepSeekChat:
    def __init__(self, model="deepseek-r1:1.5b"):
        self.model = model
        self.history = []  # Store conversation history

    def chat(self, prompt):
        self.history.append({"role": "user", "content": prompt})
        response = ollama.chat(model=self.model, messages=self.history)
        
        bot_response = response["message"]["content"]
        self.history.append({"role": "assistant", "content": bot_response})
        
        return bot_response

    def clear_history(self):
        """Clears chat memory."""
        self.history = []
