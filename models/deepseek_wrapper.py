import ollama

def deepseek_chat(prompt, history=[]):
    response = ollama.chat(model="deepseek", messages=history + [{"role": "user", "content": prompt}])
    return response["message"]["content"]
