import ollama

# Preload model on app start
MODEL_NAME = "deepseek-r1:1.5b"
ollama.pull(MODEL_NAME)

def deepseek_chat(prompt, history):
    """Chat with DeepSeek using Ollama API."""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=history + [{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]
