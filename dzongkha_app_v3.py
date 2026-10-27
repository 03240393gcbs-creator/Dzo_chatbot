"""Dzongkha chatbot v3 (verified answers + Dzongkha spell/structure checking): ask in English or Dzongkha, answered in Dzongkha.

Modes
  Chat (translate pipeline): Dzongkha->English, answer in English, English->Dzongkha
  Chat (fine-tuned)        : your LoRA model (only knows what is in your small dataset)
  Summarize                : summary written in English, then translated to Dzongkha
  Next word                : GPT-2 next-word suggestions

Run from the project folder:
    pip install transformers sentencepiece torch
    streamlit run app.py
"""
import re
from pathlib import Path
import streamlit as st

from dzongkha_nlp import AnswerBank, Lexicon, analyse, normalize

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
                           max_new_tokens=max_new, num_beams=5,
                           no_repeat_ngram_size=3, repetition_penalty=1.2, early_stopping=True)
    return tok.batch_decode(out, skip_special_tokens=True)[0].strip()


def en_to_dz(text, repo):
    """Translate sentence by sentence, then clean each sentence (spaces, shad, wrong
    spellings, locative ལུ) so the joined result reads as proper Dzongkha."""
    parts = [normalize(translate(s, repo, EN, DZ)) for s in split_sentences(text)]
    return normalize(" ".join(p for p in parts if p))


# ---------------------------------------------------- verified bank + spell check
@st.cache_resource
def load_bank_and_lexicon():
    bank = AnswerBank("verified_kb.json")
    lex = Lexicon()
    bank.build_lexicon(lex)
    lex.add_file("data/lexicon.txt")
    lex.add_file("data/extra_corpus.txt")      # optional: put real Dzongkha text here (one sentence per line)
    return bank, lex


def pipeline_chat(en_history, user_text, dz_repo, max_tokens, temp):
    """Return (dz_reply, english_question, english_answer, source).
    source is "verified" when the answer came from verified_kb.json."""
    bank, _ = load_bank_and_lexicon()
    item, score = bank.lookup(user_text)
    if item:
        return item["dz_answer"], user_text, "(verified answer bank)", "verified"
    if dzongkha_ratio(user_text) >= 0.5:
        en_q = translate(user_text, DZ_TO_EN, DZ, EN)
    else:
        en_q = user_text
    en_history.append({"role": "user", "content": en_q})
    en_a = clean_en(run_llm(load_base, EN_SYSTEM, en_history, max_tokens, temp))
    en_history.append({"role": "assistant", "content": en_a})
    return en_to_dz(en_a, dz_repo), en_q, en_a, "machine"


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


# ======================================================================== UI
# Front end only. Everything above this line (models, translation, answer bank,
# spell checking) is unchanged back-end logic.
import html as _html

APP_NAME = "Dzongkha AI Assistant"
MODES = {
    "💬  Chat": "Chat (translate pipeline)",
    "🎓  Chat (fine-tuned)": "Chat (fine-tuned)",
    "📝  Summarize": "Summarize",
    "✨  Next word": "Next word",
}
SUGGESTIONS = [
    ("👋 Greeting", "Hello"),
    ("🙏 Say thank you", "How do you say thank you in Dzongkha?"),
    ("🏛️ Capital of Bhutan", "What is the capital of Bhutan?"),
    ("📅 Days of the week", "How do you say Tuesday in Dzongkha?"),
    ("🏔️ Tiger's Nest", "What is Paro Taktsang?"),
    ("🔢 Numbers", "How do you say 5 in Dzongkha?"),
]

st.set_page_config(page_title=APP_NAME, page_icon="🐉", layout="centered",
                   initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Serif+Tibetan:wght@400;600&display=swap');
:root{
  --brand:#b4232a; --brand-dark:#7f1519; --gold:#e3a72f; --ink:#1f2430; --muted:#6b7280;
  --card:#ffffff; --bg:#f6f4f1; --line:#e7e2dc; --user:#fdecea; --ok:#12805c; --warn:#b45309;
}
html, body, [class*="css"], .stApp { font-family:'Inter',system-ui,sans-serif; }
.stApp { background: var(--bg); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility:hidden; height:0; }
.block-container { padding-top:1.2rem; padding-bottom:6rem; max-width:860px; }

/* Dzongkha text */
.dz, .dz * { font-family:'Noto Serif Tibetan','Jomolhari','Microsoft Himalaya','Kailasa',serif !important;
             font-size:1.25rem; line-height:2.0; color:var(--ink); }

/* header banner */
.hero { background:linear-gradient(135deg,var(--brand-dark),var(--brand) 60%,#d9482f);
        border-radius:20px; padding:22px 26px; color:#fff; display:flex; align-items:center; gap:16px;
        box-shadow:0 8px 24px rgba(127,21,25,.25); margin-bottom:18px; }
.hero .logo { font-size:2.4rem; background:rgba(255,255,255,.15); border-radius:16px; width:62px; height:62px;
              display:flex; align-items:center; justify-content:center; }
.hero h1 { margin:0; font-size:1.55rem; font-weight:700; color:#fff; letter-spacing:-.01em; }
.hero p  { margin:2px 0 0; opacity:.88; font-size:.93rem; color:#fff; }
.hero .tib { font-family:'Noto Serif Tibetan',serif; font-size:1.05rem; color:var(--gold); margin-top:4px; }

/* welcome */
.welcome { text-align:center; padding:26px 8px 6px; }
.welcome h2 { font-size:1.4rem; font-weight:600; color:var(--ink); margin-bottom:4px; }
.welcome p  { color:var(--muted); margin-bottom:0; }

/* suggestion chips */
div[data-testid="stButton"] > button { border-radius:14px; border:1px solid var(--line); background:var(--card);
    color:var(--ink); font-weight:500; padding:.65rem .9rem; transition:all .15s ease; width:100%; }
div[data-testid="stButton"] > button:hover { border-color:var(--brand); color:var(--brand);
    transform:translateY(-1px); box-shadow:0 4px 12px rgba(0,0,0,.07); }

/* chat bubbles */
[data-testid="stChatMessage"] { background:var(--card); border:1px solid var(--line); border-radius:18px;
    padding:14px 18px; margin-bottom:12px; box-shadow:0 1px 3px rgba(0,0,0,.04); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background:var(--user); border-color:#f3cfcb; }

/* badges */
.badge { display:inline-flex; align-items:center; gap:6px; font-size:.74rem; font-weight:600; padding:3px 10px;
         border-radius:999px; margin-top:6px; }
.badge.ok   { background:#e3f6ee; color:var(--ok); }
.badge.warn { background:#fdf0dc; color:var(--warn); }

/* chat input */
[data-testid="stChatInput"] { border-radius:16px; border:1px solid var(--line); box-shadow:0 4px 18px rgba(0,0,0,.06); }
[data-testid="stChatInput"] textarea { font-family:'Inter','Noto Serif Tibetan',sans-serif; }

/* sidebar */
section[data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
.brand { display:flex; align-items:center; gap:10px; margin:4px 0 14px; }
.brand .b { font-size:1.6rem; }
.brand .t { font-weight:700; font-size:1.05rem; color:var(--ink); line-height:1.1; }
.brand .s { font-size:.75rem; color:var(--muted); }
.side-label { font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin:14px 0 4px; font-weight:600; }

/* result cards */
.card { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:16px 20px; margin:8px 0 12px; }
.card h4 { margin:0 0 8px; font-size:.8rem; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
.pred { display:flex; align-items:center; gap:12px; padding:8px 0; border-bottom:1px dashed var(--line); }
.pred:last-child { border-bottom:none; }
.pred .rk { width:26px; height:26px; border-radius:50%; background:var(--brand); color:#fff; font-size:.78rem;
            display:flex; align-items:center; justify-content:center; font-weight:600; }
.pred .bar { flex:1; height:8px; background:#f0ebe5; border-radius:99px; overflow:hidden; }
.pred .bar > i { display:block; height:100%; background:linear-gradient(90deg,var(--gold),var(--brand)); }
.pred .pc { color:var(--muted); font-size:.82rem; width:52px; text-align:right; }
.foot { text-align:center; color:var(--muted); font-size:.75rem; margin-top:18px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------- UI helpers
def dz_html(text: str) -> str:
    return f'<div class="dz">{_html.escape(text)}</div>'


def check_notes(text, lex):
    """Run the spelling/structure checker. Returns (cleaned_text, [note strings])."""
    rep = analyse(text, lex)
    notes = []
    if rep.changes:
        notes.append("Auto-fixed: " + ", ".join(f"{a} → {b}" for a, b in rep.changes))
    if rep.malformed:
        notes.append("Malformed syllables: " + ", ".join(rep.malformed))
    if rep.unknown:
        notes.append("Not in word list (check spelling): " + ", ".join(
            w + (f" (did you mean {'/'.join(s)}?)" if s else "") for w, s in rep.unknown[:8]))
    notes += rep.structure
    return rep.fixed, notes


def render_assistant(content, source=None, notes=None, debug=None):
    st.markdown(dz_html(content) if TIBETAN.search(content) else content, unsafe_allow_html=True)
    if source == "verified":
        st.markdown('<span class="badge ok">✔ Verified answer</span>', unsafe_allow_html=True)
    elif source == "machine":
        st.markdown('<span class="badge warn">⚠ Machine translated · please double-check spelling</span>',
                    unsafe_allow_html=True)
    if notes:
        with st.expander("Spelling & sentence checks"):
            for n in notes:
                st.write("• " + n)
    if debug and source != "verified":
        with st.expander("Show English steps"):
            st.write("**Question:** " + debug[0])
            st.write("**Answer:** " + debug[1])


def render_history(key):
    for m in st.session_state[key]:
        with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🐉"):
            if m["role"] == "user":
                st.write(m["content"])
            else:
                render_assistant(m["content"], m.get("source"), m.get("notes"), m.get("debug"))


def get_prompt(placeholder):
    """Typed message, or a suggestion chip that was clicked."""
    typed = st.chat_input(placeholder)
    return typed or st.session_state.pop("pending", None)


def welcome():
    st.markdown(
        '<div class="welcome"><h2>How can I help you today?</h2>'
        '<p>Ask in English or Dzongkha. Try one of these:</p></div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (label, prompt) in enumerate(SUGGESTIONS):
        if cols[i % 2].button(label, key=f"sg{i}", use_container_width=True):
            st.session_state["pending"] = prompt
            st.rerun()


# -------------------------------------------------------------------- state
for key, default in [("hist_pipe", []), ("en_hist", []), ("hist_ft", [])]:
    st.session_state.setdefault(key, default)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown('<div class="brand"><div class="b">🐉</div><div><div class="t">Dzongkha AI</div>'
                '<div class="s">English ⇄ རྫོང་ཁ</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Mode</div>', unsafe_allow_html=True)
    mode = MODES[st.radio("Mode", list(MODES), label_visibility="collapsed")]

    with st.expander("⚙️ Settings"):
        max_tokens = st.slider("Max reply length", 64, 1024, 200, 32)
        temp = st.slider("Creativity", 0.0, 1.0, 0.3, 0.1)
        dz_choice = st.selectbox("English → Dzongkha model", list(EN_TO_DZ_OPTIONS))
    dz_repo = EN_TO_DZ_OPTIONS[dz_choice]

    if st.button("🗑️  New chat", use_container_width=True):
        st.session_state.hist_pipe, st.session_state.en_hist, st.session_state.hist_ft = [], [], []
        st.session_state.pop("pending", None)
        st.rerun()

    st.markdown('<div class="side-label">About</div>', unsafe_allow_html=True)
    st.caption("✔ Verified = answer from the hand-checked answer bank.\n\n"
               "⚠ Machine translated = cleaned and spell-checked automatically, but not guaranteed.")

# -------------------------------------------------------------------- header
st.markdown(
    f'<div class="hero"><div class="logo">🐉</div><div><h1>{APP_NAME}</h1>'
    '<p>Ask in English or Dzongkha · answers in Dzongkha</p>'
    '<div class="tib">འབྲུག་རྒྱལ་ཁབ་ཀྱི་རྒྱལ་ཡོངས་སྐད་ཡིག</div></div></div>', unsafe_allow_html=True)

# --------------------------------------------------------------------- modes
if mode == "Chat (translate pipeline)":
    if not st.session_state.hist_pipe:
        welcome()
    render_history("hist_pipe")
    if q := get_prompt("Message the assistant (English or Dzongkha)…"):
        st.session_state.hist_pipe.append({"role": "user", "content": q})
        with st.chat_message("user", avatar="🧑"):
            st.write(q)
        with st.chat_message("assistant", avatar="🐉"):
            with st.spinner("Thinking…"):
                try:
                    dz, en_q, en_a, source = pipeline_chat(st.session_state.en_hist, q, dz_repo, max_tokens, temp)
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
                    st.stop()
            notes = []
            if source != "verified":
                dz, notes = check_notes(dz, load_bank_and_lexicon()[1])
            render_assistant(dz, source, notes, (en_q, en_a))
        st.session_state.hist_pipe.append({"role": "assistant", "content": dz, "source": source,
                                           "notes": notes, "debug": (en_q, en_a)})

elif mode == "Chat (fine-tuned)":
    st.info("This is your fine-tuned LoRA model. It only knows what is in your training data.", icon="🎓")
    if not st.session_state.hist_ft:
        welcome()
    render_history("hist_ft")
    if q := get_prompt("Message the fine-tuned model…"):
        st.session_state.hist_ft.append({"role": "user", "content": q})
        with st.chat_message("user", avatar="🧑"):
            st.write(q)
        with st.chat_message("assistant", avatar="🐉"):
            with st.spinner("Thinking…"):
                reply, ok = dzongkha_only(SYSTEM, st.session_state.hist_ft, max_tokens, temp)
            reply, notes = check_notes(reply, load_bank_and_lexicon()[1])
            if not ok:
                notes.insert(0, "The model did not produce clean Dzongkha.")
            render_assistant(reply, "machine", notes)
        st.session_state.hist_ft.append({"role": "assistant", "content": reply, "source": "machine", "notes": notes})

elif mode == "Summarize":
    st.markdown("#### 📝 Summarize text")
    text = st.text_area("Paste text (English or Dzongkha)", height=220, placeholder="Paste or type your text here…")
    if st.button("Summarize", type="primary") and text.strip():
        with st.spinner("Summarizing…"):
            try:
                dz, en_s = pipeline_summary(text, dz_repo, max_tokens, temp)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()
        dz, notes = check_notes(dz, load_bank_and_lexicon()[1])
        st.markdown(f'<div class="card"><h4>Summary (Dzongkha)</h4>{dz_html(dz)}</div>', unsafe_allow_html=True)
        if notes:
            with st.expander("Spelling & sentence checks"):
                for n in notes:
                    st.write("• " + n)
        with st.expander("English summary"):
            st.write(en_s)

else:
    st.markdown("#### ✨ Next-word prediction")
    prefix = st.text_input("Dzongkha text", value="འབྲུག་རྒྱལ་ཁབ་")
    if st.button("Predict", type="primary") and prefix.strip():
        with st.spinner("Predicting…"):
            preds = predict_next_words(prefix)
        rows = "".join(
            f'<div class="pred"><div class="rk">{i}</div><div class="dz" style="min-width:90px">{_html.escape(w)}</div>'
            f'<div class="bar"><i style="width:{min(100, p * 100 * 2):.0f}%"></i></div><div class="pc">{p:.1%}</div></div>'
            for i, (w, p) in enumerate(preds, 1))
        st.markdown(f'<div class="card"><h4>Top suggestions</h4>{rows}</div>', unsafe_allow_html=True)

st.markdown('<div class="foot">Dzongkha AI Assistant · NLP module project</div>', unsafe_allow_html=True)