import ollama

def deepseek_chat(prompt, history=None):
    if history is None:
        history = []
    
    response = ollama.chat(
        model="deepseek-r1:1.5b",  # Ensure correct model name
        messages=history + [{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]
