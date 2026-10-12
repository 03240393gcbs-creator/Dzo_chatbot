import ollama

def ask_dzongkha_bot(english_question):
    # Set the system constraints to enforce the language rule
    system_instruction = (
        "You are an advanced cross-lingual AI assistant. "
        "The user will ask you a question in English. "
        "You must process the question, understand its meaning, "
        "and reply exclusively in the Dzongkha language script. "
        "Do not include English words in your reply."
    )
    
    response = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'system', 'content': system_instruction},
            {'role': 'user', 'content': english_question}
        ]
    )
    
    return response['message']['content']

# Test it locally on your M1 Mac
question = "What are the core ingredients needed to cook Ema Datshi?"
print(f"User Question: {question}\n")

bot_reply = ask_dzongkha_bot(question)
print(f"Bot Reply:\n{bot_reply}")