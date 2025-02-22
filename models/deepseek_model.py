import re
import ollama

MODEL_NAME = "deepseek-r1:1.5b"

def deepseek_chat(prompt, history):
    """Generate a chat response using Ollama's DeepSeek model."""
    try:
        response = ollama.chat(model=MODEL_NAME, messages=history + [{"role": "user", "content": prompt}])

        if not response or "message" not in response:
            return "Sorry, I couldn't generate a response. Try rephrasing your question."

        cleaned_response = re.sub(r"<think>.*?</think>", "", response["message"]["content"], flags=re.DOTALL).strip()
        return cleaned_response

    except Exception as e:
        return f"Error: {str(e)}"