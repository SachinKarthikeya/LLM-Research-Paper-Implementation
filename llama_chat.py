import ollama

model_name = "llama3.2:1b"
prompt = input("Enter your prompt: ")

response = ollama.chat(
    model=model_name,
    messages=[{"role": "user", "content": prompt}]
)

print("AI Response:", response["message"]["content"])