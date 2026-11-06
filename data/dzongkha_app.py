"""Dzongkha chatbot: ask in English or Dzongkha, always answered in Dzongkha.

Run from the project folder (the one containing data/):
    streamlit run dzongkha_app.py
"""
import re
import streamlit as st

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "./data/adapters"   # where your LoRA adapters were saved

SYSTEM = "Always reply in Dzongkha."
SUMMARY_SYSTEM = "Always reply in Dzongkha. Summarize the text the user gives you briefly in Dzongkha."
RETRY_SYSTEM = ("You must answer ONLY in Dzongkha, written in Tibetan script. "
                "Do not use English or Chinese.")

TIBETAN = re.compile(r"[\u0F00-\u0FFF]")
LETTERS = re.compile(r"[A-Za-z\u4e00-\u9fff\u0F00-\u0FFF]")


def dzongkha_ratio(text: str) -> float:
    """Share of letters that are Tibetan script (Dzongkha uses it)."""
    letters = LETTERS.findall(text)
    if not letters:
        return 0.0
    return len(TIBETAN.findall(text)) / len(letters)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    from mlx_lm import load
    return load(BASE_MODEL, adapter_path=ADAPTER_PATH)


def generate_reply(system: str, history: list, max_tokens: int, temp: float) -> str:
    from mlx_lm import generate
    model, tokenizer = load_model()
    messages = [{"role": "system", "content": system}] + history
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kwargs = {}
    try:
        from mlx_lm.sample_utils import make_sampler
        kwargs["sampler"] = make_sampler(temp=temp)
    except Exception:
        pass
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False, **kwargs).strip()


def dzongkha_only(system: str, history: list, max_tokens: int, temp: float):
    """Generate, and retry with a stricter prompt if the reply is not Dzongkha."""
    reply = generate_reply(system, history, max_tokens, temp)
    if dzongkha_ratio(reply) >= 0.6:
        return reply, True
    reply = generate_reply(RETRY_SYSTEM, history, max_tokens, temp=0.2)
    return reply, dzongkha_ratio(reply) >= 0.6


st.set_page_config(page_title="Dzongkha Chatbot", page_icon="🐉")
st.title("Dzongkha Chatbot")
st.caption("Ask in English or Dzongkha. Answers come back in Dzongkha.")

with st.sidebar:
    mode = st.radio("Mode", ["Chat", "Summarize"])
    max_tokens = st.slider("Max reply length", 64, 1024, 300, 32)
    temp = st.slider("Creativity", 0.0, 1.0, 0.3, 0.1)
    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()

st.session_state.setdefault("history", [])

if mode == "Chat":
    for m in st.session_state.history:
        with st.chat_message(m["role"]):
            st.write(m["content"])
    if q := st.chat_input("Ask a question (English or Dzongkha)"):
        st.session_state.history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply, ok = dzongkha_only(SYSTEM, st.session_state.history, max_tokens, temp)
            st.write(reply)
            if not ok:
                st.warning("The model did not produce clean Dzongkha. It likely needs more training data.")
        st.session_state.history.append({"role": "assistant", "content": reply})
else:
    text = st.text_area("Paste text to summarize (English or Dzongkha)", height=250)
    if st.button("Summarize") and text.strip():
        with st.spinner("Summarizing..."):
            reply, ok = dzongkha_only(SUMMARY_SYSTEM, [{"role": "user", "content": text}], max_tokens, temp)
        st.subheader("Summary (Dzongkha)")
        st.write(reply)
        if not ok:
            st.warning("The model did not produce clean Dzongkha. It likely needs more training data.")