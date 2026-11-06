"""Dzongkha chatbot v2: ask in English or Dzongkha, answered in Dzongkha.

Modes
  Chat (translate pipeline): Dzongkha->English, answer in English, English->Dzongkha
  Chat (fine-tuned)        : your LoRA model (only knows what is in your small dataset)
  Summarize                : summary written in English, then translated to Dzongkha
  Next word                : GPT-2 next-word suggestions

Run from the project folder:
    pip install transformers sentencepiece torch
    streamlit run dzongkha_app_v2.py
"""
import re
import streamlit as st

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "./data/adapters"
NEXTWORD_REPO = "KarmaCST/dzongkha_nextword_gpt2"

DZ_TO_EN = "KarmaCST/nllb-200-distilled-600M-dz-to-en"
EN_TO_DZ_OPTIONS = {
    "KarmaCST en->dz (73K)": "KarmaCST/nllb-200-distilled-600M-73K-en-to-dz",
    "KarmaCST en->dz": "KarmaCST/nllb-200-distilled-600M-en-to-dz",
    "Meta NLLB-200 base": "facebook/nllb-200-distilled-600M",
    "My fine-tuned (./models/nllb-en2dz)": "./models/nllb-en2dz",
}
DZ, EN = "dzo_Tibt", "eng_Latn"

SYSTEM = "Always reply in Dzongkha."
SUMMARY_SYSTEM = "Always reply in Dzongkha. Summarize the text the user gives you briefly in Dzongkha."
RETRY_SYSTEM = ("You must answer ONLY in Dzongkha, written in Tibetan script. "
                "Do not use English or Chinese.")
EN_SYSTEM = ("You are a helpful assistant. Answer in at most 3 short, simple English sentences. "
             "No lists, no markdown, no emojis.")
EN_SUMMARY_SYSTEM = "Summarize the user's text in 2 short, simple English sentences. No lists, no markdown."

TIBETAN = re.compile(r"[\u0F00-\u0FFF]")
LETTERS = re.compile(r"[A-Za-z\u4e00-\u9fff\u0F00-\u0FFF]")


def dzongkha_ratio(text: str) -> float:
    letters = LETTERS.findall(text)
    return len(TIBETAN.findall(text)) / len(letters) if letters else 0.0


def clean_en(text: str) -> str:
    text = re.sub(r"[*#`_>]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str, limit: int = 4):
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", clean_en(text)) if p.strip()]
    return parts[:limit]


# ---------------------------------------------------------------- LLM (MLX)
@st.cache_resource(show_spinner="Loading English model...")
def load_base():
    from mlx_lm import load
    return load(BASE_MODEL)


@st.cache_resource(show_spinner="Loading fine-tuned model...")
def load_finetuned():
    from mlx_lm import load
    return load(BASE_MODEL, adapter_path=ADAPTER_PATH)


def run_llm(loader, system, history, max_tokens, temp):
    from mlx_lm import generate
    model, tokenizer = loader()
    messages = [{"role": "system", "content": system}] + history
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kwargs = {}
    try:
        from mlx_lm.sample_utils import make_sampler
        kwargs["sampler"] = make_sampler(temp=temp)
    except Exception:
        pass
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False, **kwargs).strip()


def dzongkha_only(system, history, max_tokens, temp):
    reply = run_llm(load_finetuned, system, history, max_tokens, temp)
    if dzongkha_ratio(reply) >= 0.6:
        return reply, True
    reply = run_llm(load_finetuned, RETRY_SYSTEM, history, max_tokens, 0.2)
    return reply, dzongkha_ratio(reply) >= 0.6


# ------------------------------------------------------- Translation (NLLB)
@st.cache_resource(show_spinner="Loading translation model (first time downloads ~2.5 GB)...")
def load_translator(repo, src_lang):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained(repo, src_lang=src_lang)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(repo)
    mdl.eval()
    return tok, mdl


def translate(text, repo, src, tgt, max_new=96):
    import torch
    tok, mdl = load_translator(repo, src)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = mdl.generate(**inputs, forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
                           max_new_tokens=max_new, num_beams=4)
    return tok.batch_decode(out, skip_special_tokens=True)[0].strip()


def en_to_dz(text, repo):
    """Translate sentence by sentence (the models were trained on short sentences)."""
    return " ".join(translate(s, repo, EN, DZ) for s in split_sentences(text))


def pipeline_chat(en_history, user_text, dz_repo, max_tokens, temp):
    """Return (dz_reply, english_question, english_answer)."""
    if dzongkha_ratio(user_text) >= 0.5:
        en_q = translate(user_text, DZ_TO_EN, DZ, EN)
    else:
        en_q = user_text
    en_history.append({"role": "user", "content": en_q})
    en_a = clean_en(run_llm(load_base, EN_SYSTEM, en_history, max_tokens, temp))
    en_history.append({"role": "assistant", "content": en_a})
    return en_to_dz(en_a, dz_repo), en_q, en_a


def pipeline_summary(text, dz_repo, max_tokens, temp):
    en_text = translate(text, DZ_TO_EN, DZ, EN) if dzongkha_ratio(text) >= 0.5 else text
    en_s = clean_en(run_llm(load_base, EN_SUMMARY_SYSTEM, [{"role": "user", "content": en_text}], max_tokens, temp))
    return en_to_dz(en_s, dz_repo), en_s


# ------------------------------------------------------- Next word (GPT-2)
@st.cache_resource(show_spinner="Loading next-word model (first time downloads ~3 GB)...")
def load_nextword():
    import torch
    from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = PreTrainedTokenizerFast.from_pretrained(NEXTWORD_REPO)
    mdl = GPT2LMHeadModel.from_pretrained(NEXTWORD_REPO).to(device)
    mdl.eval()
    return mdl, tok, device


def predict_next_words(text, k=5):
    import torch
    mdl, tok, device = load_nextword()
    ids = tok.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(mdl(ids).logits[:, -1, :], dim=-1)
        top_p, top_i = torch.topk(probs, 50)
    out = []
    for i, p in zip(top_i[0], top_p[0]):
        w = tok.decode([int(i)]).strip()
        if w and not re.search(r"<0x[0-9A-Fa-f]+>", w):
            out.append((w, p.item()))
        if len(out) == k:
            break
    return out


# ----------------------------------------------------------------------- UI
st.set_page_config(page_title="Dzongkha Chatbot", page_icon="🐉")
st.title("Dzongkha Chatbot")
st.caption("Ask in English or Dzongkha. Answers come back in Dzongkha.")

for key, default in [("hist_pipe", []), ("en_hist", []), ("hist_ft", [])]:
    st.session_state.setdefault(key, default)

with st.sidebar:
    mode = st.radio("Mode", ["Chat (translate pipeline)", "Chat (fine-tuned)", "Summarize", "Next word"])
    max_tokens = st.slider("Max reply length", 64, 1024, 200, 32)
    temp = st.slider("Creativity", 0.0, 1.0, 0.3, 0.1)
    dz_choice = st.selectbox("English -> Dzongkha model", list(EN_TO_DZ_OPTIONS))
    dz_repo = EN_TO_DZ_OPTIONS[dz_choice]
    if st.button("Clear chat"):
        st.session_state.hist_pipe, st.session_state.en_hist, st.session_state.hist_ft = [], [], []
        st.rerun()

if mode == "Chat (translate pipeline)":
    st.info("Pipeline: Dzongkha -> English -> answer -> Dzongkha. "
            "The English->Dzongkha step is the weakest part, so expect imperfect Dzongkha.")
    for m in st.session_state.hist_pipe:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m.get("debug"):
                with st.expander("Show English steps"):
                    st.write("**Question (English):** " + m["debug"][0])
                    st.write("**Answer (English):** " + m["debug"][1])
    if q := st.chat_input("Ask a question (English or Dzongkha)"):
        st.session_state.hist_pipe.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("Translating and thinking..."):
                try:
                    dz, en_q, en_a = pipeline_chat(st.session_state.en_hist, q, dz_repo, max_tokens, temp)
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.stop()
            st.write(dz)
            with st.expander("Show English steps"):
                st.write("**Question (English):** " + en_q)
                st.write("**Answer (English):** " + en_a)
        st.session_state.hist_pipe.append({"role": "assistant", "content": dz, "debug": (en_q, en_a)})

elif mode == "Chat (fine-tuned)":
    st.info("This is your LoRA model. It only knows what is in your small training set.")
    for m in st.session_state.hist_ft:
        with st.chat_message(m["role"]):
            st.write(m["content"])
    if q := st.chat_input("Ask a question (English or Dzongkha)"):
        st.session_state.hist_ft.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply, ok = dzongkha_only(SYSTEM, st.session_state.hist_ft, max_tokens, temp)
            st.write(reply)
            if not ok:
                st.warning("The model did not produce clean Dzongkha.")
        st.session_state.hist_ft.append({"role": "assistant", "content": reply})

elif mode == "Summarize":
    text = st.text_area("Paste text to summarize (English or Dzongkha)", height=250)
    if st.button("Summarize") and text.strip():
        with st.spinner("Summarizing..."):
            try:
                dz, en_s = pipeline_summary(text, dz_repo, max_tokens, temp)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()
        st.subheader("Summary (Dzongkha)")
        st.write(dz)
        with st.expander("Show English summary"):
            st.write(en_s)

else:
    st.subheader("Dzongkha next-word prediction")
    prefix = st.text_input("Dzongkha text", value="འབྲུག་རྒྱལ་ཁབ་")
    if st.button("Predict") and prefix.strip():
        with st.spinner("Predicting..."):
            preds = predict_next_words(prefix)
        for rank, (w, p) in enumerate(preds, 1):
            st.write(f"{rank}. **{w}**  ({p:.1%})")