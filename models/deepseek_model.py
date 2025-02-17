import ollama
import re

# Preload model on app start
MODEL_NAME = "deepseek-r1:1.5b"
ollama.pull(MODEL_NAME)

def deepseek_chat(prompt, history):
    response = ollama.chat(
        model="deepseek-r1:1.5b",
        messages=history + [{"role": "user", "content": prompt}]
    )

    # Ensure the response doesn't contain internal reasoning (`<think>`)
    cleaned_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    return cleaned_response
