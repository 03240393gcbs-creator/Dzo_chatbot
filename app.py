import ollama
import streamlit as st

SYSTEM_INSTRUCTION = (
    "You are an advanced cross-lingual AI assistant. "
    "The user will ask you a question in English. "
    "You must process the question, understand its meaning, "
    "and reply exclusively in the Dzongkha language script. "
    "Do not include English words in your reply."
)

def ask_dzongkha_bot(history):
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "system", "content": SYSTEM_INSTRUCTION}] + history,
    )
    return response["message"]["content"]

st.set_page_config(page_title="Dzongkha Chatbot", page_icon="🇧🇹")
st.title("🇧🇹 Dzongkha Chatbot")
st.caption("Ask in English, get the reply in Dzongkha.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask your question in English..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = ask_dzongkha_bot(st.session_state.messages)
        st.write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
