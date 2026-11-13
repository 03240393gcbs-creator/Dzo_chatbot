"""Dzongkha chatbot v3 (verified answers + Dzongkha spell/structure checking): ask in English or Dzongkha, answered in Dzongkha.



Modes

  Chat (translate pipeline): Dzongkha->English, answer in English, English->Dzongkha

  Chat (fine-tuned)        : your LoRA model (only knows what is in your small dataset)

  Summarize                : summary written in English, then translated to Dzongkha

  Next word                : GPT-2 next-word suggestions



Run from the project folder:

    pip install transformers sentencepiece torch

    streamlit run dzongkha_app_v3.py

"""

import json
import os

import re

import time

import urllib.parse

import urllib.request

from functools import lru_cache

from pathlib import Path

import streamlit as st



from dzongkha_nlp import AnswerBank, FactBank, Lexicon, analyse, normalize

from govtech_dnlp import segment_dzongkha



# Bigger model = better, more accurate answers (slower, larger first download).

#   small & fast : "Qwen/Qwen2.5-1.5B-Instruct"

#   recommended  : "mlx-community/Qwen2.5-3B-Instruct-4bit"

#   best quality : "mlx-community/Qwen2.5-7B-Instruct-4bit"   (needs 16 GB RAM)

BASE_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"

# Faster option: Ollama (https://ollama.com). Install it, then run once in Terminal:   ollama pull qwen2.5:3b

# If Ollama is running, the app uses it automatically; otherwise it falls back to the MLX model above.

OLLAMA_URL = "http://localhost:11434"

OLLAMA_MODEL = "qwen2.5:3b"          # other choices: "llama3.2:3b" (faster), "qwen2.5:7b" (better, slower)

USE_WEB = False                        # look things up on the web before answering general questions

TIMES = {}                            # seconds spent per step, shown under each reply

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

EN_SYSTEM = ("In this app, dzo means the Dzongkha language (ISO 639-3 code dzo), not an animal. "

             "You are a helpful, knowledgeable assistant. Answer ANY question about any country or topic in at most 3 short, "

             "simple English sentences. No lists, no markdown, no emojis. "

             "Use the following background ONLY when the question is about Bhutan; for other countries answer normally. Bhutan is a constitutional monarchy. Its head of state is the King (Jigme Khesar Namgyel Wangchuck) "

             "and its Prime Minister is Tshering Tobgay. Bhutan has no president. Its capital is Thimphu. "

             "If you are not sure of a fact, say you are not sure instead of guessing.")

GROUNDED_SYSTEM = ("You are a helpful assistant for a Bhutan/Dzongkha app. Answer using ONLY the facts below. "

                   "Do not add any information that is not in the facts. Answer in at most 3 short, simple English sentences.\n\nFacts:\n{facts}")

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





# ---------------------------------------------------------------- live web lookup

SKIP_WEB = re.compile(r"^\s*(hi|hello|hey|thanks?|thank you|bye|goodbye|good (morning|afternoon|evening|night)|how are you|who are you|what is your name)\b", re.I)





def _wikipedia(query, n=3):

    url = ("https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrlimit=%d&prop=extracts"

           "&exintro=1&explaintext=1&exsentences=4&format=json&gsrsearch=%s" % (n, urllib.parse.quote(query)))

    req = urllib.request.Request(url, headers={"User-Agent": "DzongkhaChatbot/1.0 (student project)"})

    with urllib.request.urlopen(req, timeout=6) as r:

        pages = json.loads(r.read()).get("query", {}).get("pages", {})

    return [p.get("extract", "") for p in sorted(pages.values(), key=lambda p: p.get("index", 99))]





@lru_cache(maxsize=128)

def web_context(query: str) -> str:

    """Up-to-date snippets from the web (DuckDuckGo if the 'ddgs' package is installed, else Wikipedia)."""

    snippets = []

    try:

        try:

            from ddgs import DDGS

        except ImportError:

            from duckduckgo_search import DDGS

        with DDGS() as d:

            snippets = [r.get("body", "") for r in d.text(query, max_results=5)]

    except Exception:

        snippets = []

    if not any(snippets):

        try:

            snippets = _wikipedia(query)

        except Exception:

            snippets = []

    text = "\n".join("- " + x.strip().replace("\n", " ") for x in snippets if x and x.strip())

    return text[:1800]





@lru_cache(maxsize=1)

def ollama_ready():

    """True if Ollama is running locally and has OLLAMA_MODEL installed."""

    try:

        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=1) as r:

            names = [m.get("name", "") for m in json.loads(r.read()).get("models", [])]

        return OLLAMA_MODEL in names

    except Exception:

        return False





def run_ollama(system, history, max_tokens, temp):

    body = json.dumps({"model": OLLAMA_MODEL, "stream": False, "keep_alive": "30m",

                       "messages": [{"role": "system", "content": system}] + history,

                       "options": {"temperature": temp, "num_predict": max_tokens}}).encode()

    req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=body, headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(req, timeout=90) as r:

        return json.loads(r.read())["message"]["content"].strip()





def run_llm(loader, system, history, max_tokens, temp):

    t0 = time.time()

    try:

        if loader is load_base and ollama_ready():

            return run_ollama(system, history, max_tokens, temp)

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

    finally:

        TIMES["model"] = TIMES.get("model", 0) + time.time() - t0





def dzongkha_only(system, history, max_tokens, temp):

    reply = run_llm(load_finetuned, system, history, max_tokens, temp)

    if dzongkha_ratio(reply) >= 0.6:

        return reply, True

    reply = run_llm(load_finetuned, RETRY_SYSTEM, history, max_tokens, 0.2)

    return reply, dzongkha_ratio(reply) >= 0.6





# ------------------------------------------------------- Translation (NLLB)

@st.cache_resource(show_spinner="Loading Dzongkha translation model (first use)...")
def load_translator(repo, src_lang):
    """Load each translator only once. Prefer Apple M1 GPU where supported.

    Set DZ_TRANSLATION_DEVICE=cpu to compare CPU performance if MPS is slower.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    requested = os.environ.get("DZ_TRANSLATION_DEVICE", "auto").lower()
    if requested == "cpu":
        device = "cpu"
    elif requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable. Set DZ_TRANSLATION_DEVICE=cpu.")
        device = "mps"
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    if device == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 4))

    tok = AutoTokenizer.from_pretrained(repo, src_lang=src_lang)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(repo, low_cpu_mem_usage=False)
    mdl.eval()
    try:
        mdl = mdl.to(device)
    except (RuntimeError, NotImplementedError) as exc:
        if requested == "mps":
            raise
        st.warning(f"Apple GPU translation unavailable; using CPU instead ({exc}).")
        device = "cpu"
        torch.set_num_threads(min(4, os.cpu_count() or 4))
        mdl = mdl.to(device)
    return tok, mdl, device


# Every translation is saved to translation_cache.json, so the same sentence is instant next time.

CACHE_FILE = Path(__file__).resolve().parent / "translation_cache.json"

_CACHE = None





def _cache():

    global _CACHE

    if _CACHE is None:

        try:

            _CACHE = json.loads(CACHE_FILE.read_text(encoding="utf-8"))

        except Exception:

            _CACHE = {}

    return _CACHE





def translate(text, repo, src, tgt, max_new=48):

    t0 = time.time()

    try:

        return _translate(text, repo, src, tgt, max_new)

    finally:

        TIMES["translate"] = TIMES.get("translate", 0) + time.time() - t0





def _translate(text, repo, src, tgt, max_new=48):

    import torch

    key = f"{repo}|{src}|{tgt}|{text}"

    cache = _cache()

    if key in cache:

        return cache[key]

    tok, mdl, device = load_translator(repo, src)

    inputs = tok(text, return_tensors="pt", truncation=True, max_length=96).to(device)

    with torch.inference_mode():

        out = mdl.generate(**inputs, forced_bos_token_id=tok.convert_tokens_to_ids(tgt),

                           max_new_tokens=max_new, num_beams=1, do_sample=False,
                           use_cache=True, no_repeat_ngram_size=3, repetition_penalty=1.1)

    result = tok.batch_decode(out.to("cpu"), skip_special_tokens=True)[0].strip()

    if result:

        cache[key] = result

        try:

            CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

        except Exception:

            pass

    return result





def en_to_dz(text, repo):

    """Translate sentence by sentence, then clean each sentence (spaces, shad, wrong

    spellings, locative ལུ) so the joined result reads as proper Dzongkha."""

    parts = [normalize(translate(s, repo, EN, DZ)) for s in split_sentences(text, limit=2)]

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





@st.cache_resource

def load_facts():

    return FactBank("facts_en.json")





@st.cache_resource

def training_questions():

    """English questions from your training data (used to decide when the fine-tuned LoRA model should answer)."""

    qs = []

    for f in ("data/dataset.json", "dzo_dataset_full.json"):

        try:

            for d in json.loads(Path(f).read_text(encoding="utf-8")):

                if d.get("instruction"):

                    qs.append(d["instruction"])

        except Exception:

            pass

    return qs





def in_training_data(text, threshold=0.6):

    """True if the question is (nearly) one of the training questions."""

    from dzongkha_nlp import _tok

    qt = _tok(text.lower())

    if not qt:

        return False

    for q in training_questions():

        kt = _tok(q.lower())

        if kt and len(qt & kt) / len(qt | kt) >= threshold:

            return True

    return False





def extract_translation_request(message):

    """Route explicit English-to-Dzongkha translation requests around the chat LLM.



    Returns the exact English phrase to translate, or None for normal questions.

    """

    text = message.strip()

    target = re.search(r"\b(?:dzongkha|dzongka|dzo|dz)\b", text, re.I)

    action = re.search(r"\b(?:translat(?:e|ion)|say|write|convert)\b", text, re.I)

    if not (target and action):

        return None



    # Prefer text in quotation marks: 'I love machine learning' can you translate...

    quoted = re.search(r'[“"]([^”"]{1,600})[”"]|[‘**\'**]([^’**\'**]{1,600})[’**\'**]', text)

    if quoted:

        phrase = (quoted.group(1) or quoted.group(2)).strip()

        if phrase:

            return phrase



    # "Translate I love machine learning into Dzongkha"

    match = re.search(r"\b(?:translate|convert|write|say)\s+(?:this\s+|the\s+(?:sentence|phrase)\s+)?(.+?)\s+(?:into|to|in)\s+(?:dzongkha|dzongka|dzo|dz)\b", text, re.I | re.S)

    if match:

        phrase = match.group(1).strip(" .?!:;\n\t")

        if phrase and phrase.lower() not in ("this", "it", "that"):

            return phrase



    # "Can you translate this in Dzongkha: I love machine learning"

    match = re.search(r"\b(?:dzongkha|dzongka|dzo|dz)\s*[:**\-**]\s*(.+)$", text, re.I | re.S)

    if match:

        return match.group(1).strip()



    # "Translate to Dzongkha: I love machine learning"

    match = re.search(r"\b(?:translate|convert|write|say)\b.*?\b(?:dzongkha|dzongka|dzo|dz)\s*[:**\-**]\s*(.+)$", text, re.I | re.S)

    if match:

        return match.group(1).strip()

    return None





def pipeline_chat(en_history, user_text, dz_repo, max_tokens, temp):

    """Return (dz_reply, english_question, english_answer, source).

    Order of trust:

      1. verified_kb.json  -> word-for-word Dzongkha answer         (source "verified")

      2. facts_en.json     -> English fact (no model guessing), translated to Dzongkha (source "machine")

      3. anything else     -> the language model answers (source "general")

    """

    TIMES.clear()

    # Explicit translation is a translation task, NOT a general AI question.

    # Do not ask Qwen to interpret "dzo" or generate an English answer.

    phrase = extract_translation_request(user_text)

    if phrase is not None:

        if dzongkha_ratio(phrase) >= 0.5:

            return normalize(phrase), phrase, "Direct Dzongkha text (no translation needed)", "machine"

        return normalize(translate(phrase, dz_repo, EN, DZ, max_new=64)), phrase, "Direct English-to-Dzongkha translation", "machine"

    # Instant, human-verified answer for a frequent factual question.

    # Bypass both Ollama and the slow English-to-Dzongkha translator.

    simple_q = re.sub(r"[^a-z0-9 ]", " ", user_text.lower())

    simple_q = " ".join(simple_q.split())

    if ("king" in simple_q and "bhutan" in simple_q and

            any(w in simple_q.split() for w in ("who", "name", "current"))):

        return ("འབྲུག་གི་ད་ལྟོའི་རྒྱལཔོ་འདི་ མི་དབང་འཇིགས་མེད་གེ་སར་རྣམ་རྒྱལ་དབང་ཕྱུག་མཆོག་ཨིན།",

                user_text, "(curated answer)", "verified")

    bank, _ = load_bank_and_lexicon()

    item, score = bank.lookup(user_text)

    if item:

        return item["dz_answer"], user_text, "(verified answer bank)", "verified"



    en_q = translate(user_text, DZ_TO_EN, DZ, EN) if dzongkha_ratio(user_text) >= 0.5 else user_text



    facts = load_facts().lookup(en_q)

    if facts:

        # Straight from your data - no language model involved, so it is fast and cannot invent anything.

        en_a = " ".join(f["answer_en"] for f in facts)

    else:

        # Not in your data -> let the language model answer any question (keeps the conversation going).

        ctx = ""

        if USE_WEB and not SKIP_WEB.match(en_q):

            t0 = time.time()

            ctx = web_context(en_q)

            TIMES["web"] = time.time() - t0

        system = EN_SYSTEM + "\nToday's date is " + time.strftime("%d %B %Y") + "."

        if ctx:

            system += ("\n\nWeb search results (newer than your memory - trust them over your memory, "

                       "and answer only what the question asks):\n" + ctx)

        en_history.append({"role": "user", "content": en_q})

        en_a = clean_en(run_llm(load_base, system, en_history[-8:], min(max_tokens, 120), temp))

        en_history.append({"role": "assistant", "content": en_a})

        return en_to_dz(en_a, dz_repo), en_q, en_a, ("web" if ctx else "general")

    return en_to_dz(en_a, dz_repo), en_q, en_a, "machine"





def pipeline_summary(text, dz_repo, max_tokens, temp):
    """Fast summary: one short LLM generation and ONE Dzongkha translation.

    For Dzongkha source text, an additional Dzongkha-to-English pass is needed.
    Long source text is trimmed to keep generation bounded; users should be
    informed that very long inputs may not be fully represented.
    """
    TIMES.clear()
    source = text.strip()
    # Keep input processing bounded on an M1; avoid passing enormous documents.
    source = source[:4000]
    en_text = (translate(source, DZ_TO_EN, DZ, EN, max_new=128)
               if dzongkha_ratio(source) >= 0.5 else source)
    prompt = "Summarize the following in ONE short English sentence (maximum 25 words):\n" + en_text
    en_s = clean_en(run_llm(load_base, EN_SUMMARY_SYSTEM,
                            [{"role": "user", "content": prompt}],
                            min(max_tokens, 48), temp))
    # Avoid en_to_dz(): that helper makes separate NLLB calls per sentence.
    dz = normalize(translate(en_s, dz_repo, EN, DZ, max_new=64)) if en_s else ""
    return dz, en_s





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

import base64

import html as _html

import json

import time

import uuid



import streamlit.components.v1 as components



APP_NAME = "Dzongkha AI Assistant"

BG_IMAGE = Path(__file__).resolve().parent / "assets" / "chatbot_bg.png"



# ---- Chat avatars -------------------------------------------------------------

# USER (person asking)  -> assets/user_avatar.png  (the smiling Bhutanese boy)

# BOT  (assistant)      -> assets/bot_avatar.png   (the line-art character)

# Falls back to an emoji if a file is missing.

_ASSETS = Path(__file__).resolve().parent / "assets"





def _avatar(stem: str, fallback: str) -> str:

    # looks in ./assets first, then next to this file

    here = Path(__file__).resolve().parent

    for folder in (_ASSETS, here):

        for ext in ("png", "webp", "jpg", "jpeg"):

            p = folder / f"{stem}.{ext}"

            if p.exists():

                return str(p)

    return fallback





USER_AVATAR = _avatar("user_avatar", "🧑")

BOT_AVATAR = _avatar("bot_avatar", "🐉")



# Fixed values that used to live in the sidebar "Settings" panel.

MAX_TOKENS = 80

TEMPERATURE = 0.1

DZ_REPO = EN_TO_DZ_OPTIONS["KarmaCST en->dz (73K)"]



MODES = {

    "💬  Chat": "Chat (translate pipeline)",

    "🎓  Chat (fine-tuned)": "Chat (fine-tuned)",

    "📝  Summarize": "Summarize",

    "✨  Next word": "Next word",

    "🌐  Translate": "Translate",

}

SUGGESTIONS = [

    ("👋 Greeting", "Hello"),

    ("🙏 Say thank you", "How do you say thank you in Dzongkha?"),

    ("🏛️ Capital of Bhutan", "What is the capital of Bhutan?"),

    ("📅 Days of the week", "How do you say Tuesday in Dzongkha?"),

    ("🏔️ Tiger's Nest", "What is Paro Taktsang?"),

    ("🔢 Numbers", "How do you say 5 in Dzongkha?"),

]



st.set_page_config(page_title=APP_NAME, page_icon="🐉", layout="wide",

                   initial_sidebar_state="expanded")





def _find_bg():

    here = Path(__file__).resolve().parent

    for p in (here / "assets" / "chatbot_bg.png", Path.cwd() / "assets" / "chatbot_bg.png",

              here / "chatbot_bg.png", Path.cwd() / "chatbot_bg.png"):

        if p.exists():

            return p

    return None





@st.cache_data(show_spinner=False)

def _read_b64(path_str: str, mtime: float) -> str:

    return base64.b64encode(Path(path_str).read_bytes()).decode()





def bg_data_uri() -> str:

    """The background image, embedded so Streamlit needs no static-file setup.

    A missing file is never cached, so adding it later works after a refresh."""

    p = _find_bg()

    if p is None:

        st.sidebar.warning("Background image not found. Put chatbot_bg.png in an 'assets' folder "

                           f"next to app.py (looked in: {Path(__file__).resolve().parent / 'assets'}).")

        return ""

    return "data:image/png;base64," + _read_b64(str(p), p.stat().st_mtime)





# The Dzongkha frame corners are built into this file, so no extra image files are needed.

FRAME_B64 = {

    "tl": "iVBORw0KGgoAAAANSUhEUgAAAMUAAADFCAYAAADkODbwAAADaklEQVR4nO3dUW6cSBRA0fHIe5jNZPnZzKwi+R1dNU6jKfADzvmMpTa0uSmqGuiPn//8+PXXyX78+/Pj1b+v2JarvvYqV9jG6f7+7g2AaUQBIQoIUUCIAuJz6wdbqxh7rFrxeLUtR772qtff8x5u/b5V27fi77nX3n2aYvxIMf0NXMV+zjE+CjibKIbw4docooAYH8Wk/0G/Y1sm7f8KV9ifzdWnLa926ujJ06Q3csW27H2NSfu/wlH7s+o4HD9SwLtWxSYKCFFAiAJi90T7Oz6RvNvv3HvZypGXuRzpCp9ev2KkGO6qB9aVjY/iKQfFU/bzCsZHAWcTBYQoIDZXn66wunE3e+YV5iDHMVJAiAJCFBCigBgfhQk/Zxu1gjHpYc881/iRAs4mCghRQIgC4tFRmGTzysekJc8j70gTAO8aP1JsHcwOco4yPgo4myggRAHx5US75+1/mpS/c56/9zVWbZ85CO/ajGLP10rtPeBc48RkTp8gRAEhCghRQIx6xM2Kyzzg/xo/UrjMg7ONjwLOJgoIUUCMj2LS/R48w7dMVlcc6CbaHGX8SAFnEwWEKCBEASEKiPGPuIGzGSkgRAEhCghRQIyJwiSbKTbvvHOQ8lRjRgqYQhQQooAQBYQoIJZc+2SlijsxUkCIAkIUEKKAWPINRCba3MmukcLBzxM4fYIQBYQoIEQB8TnpETcwgZECQhQQooAQBYQoIDziBsJIASEKCFFAiAJCFBCbq09bXl0rZaWKOzFSQIgCQhQQooD48lmy/51Av3Mzkgk3d3Dol8uLhCty+gQhCghRQIgCYtdNRu+uVMGVLXnAsiC4E6dPEKKAEAWEKCBc5gFhpIAQBYQoIEQBIQoAAAAAgNNsXvvkuiWeyucUEKKAEAWEKCBEASEKCFFAiAJCFBCigBAFhCggRAEhCghRQIgCQhQQooAQBYQoIEQBIQoIUUCIAkIUEKKAEAWEKCCWfLm8hzFzJ0YKCFFAiAJCFBCfX/2wE+gVk3KYbnOkeLWiZJWJJ3D6BCEKCFFAiAJCFBCigBAFhCggRAEhCghRQIgCQhQQooAQBYQoIEQBIQoIUUCIAkIUELuj8EQP7u7L5z5tEQZ35vQJQhQQooAQBYQoIEQBIQoIUUCIAkIUEKKAEAWEKCBEASEKCFFA/AYgMPxVehtJSgAAAABJRU5ErkJggg==",

    "tr": "iVBORw0KGgoAAAANSUhEUgAAAMUAAADFCAYAAADkODbwAAADaElEQVR4nO3dW07bQABAUaiyh26G5bMZVtF+VWpv4ygj/Bg753yiKDiGy2SGsfP++fPj1xtP+fj6fL/39ZnO4ZbHeNbnHvVj728IsxMFhCggRAEhCojb0qx/FkurD7Mf99vb+IrKyGs6w4rPveff8rlHLR3L9CPFGX751+B1zmP6KGBvopjETP8AfHWigJg+iqv9BT3i9cx0Dmc6liW3tZ7oDC92b6Pn5N7jRyemM/0cZvkDMHoOVxkpZvpBwHdN//YJ9iYKCFFArDLRPsN/KY8wus1h5DzOdM733p6y1rEsMVLsbKZfZu4TxSReJZYzvE5RQIgCQhQQq23zeGWj75PP8L56dluuYhkpIEQBIQoIUUCIgl2d4TIDqyD854hf3JlW5IwUEKKAEAWEKCBEwX9mmvQe4aVfPGNGbgw9eoXhTEu1Rgo2sTTanGEUEgWEKCBEATH9+zvmMTJ5Xuux9x4/emyj31MUPO0Me6JGbrB82o/3gr2JAkIUEKKAMNFmE2tt8zjik5yMFGzCNg+4EFFAiAJCFOxqpusmlkw/6eG81gjgiIm5kQJCFBCigBAFhCggrD4xhZmWao0UEKKAEAWEKCBEwRRmus5imgOBvbnFDTxJFBCigBAFhCggrD5xKe77BBsQBYQoIEQBcTv6AGAPI3urjBRc3uhmQ1FAiAJCFBCigLDNg0uxzQM2IAoIUUCIAkIUEFafeFlucQNPEgWEKCBEASEKCFfecSn3VpRcZATfJAoIUUCIAsI2Dy7hmYuL/p5wP3q8KDilLT+M3tsnCFFAiAJCFBAm2lzGo8n3va0eLjLi8pb2ONn7BN8kCghRQIgCwuoTp2SbB+xIFBCigBAFhCgAAAAAAHZj7xMvy5V38CRRQIgCQhQQooAQBYQoIEQBIQoIUUCIAkIUEKKAEAWEKCBEASEKCFFAiAJCFBCigBAFhCggRAEhCghRQIgCQhQQbrDMpazxofNGCghRQIgCQhQQt6MPAPbw8fX5z6LSowm5kYLLaxBLX/tDFBCigBAFhCggRAEhCghRQIgCQhQQooAQBYQoIEQBIQoIUUCIAkIUEKKAEAWEKCBEASEKLu3RXTuWuMEyL2vp3k9GCghRQIgCQhQQooAQBYQoIEQBIQoIUUCIAkIUEKKAEAWEKCBEAfEbIRHp0s88jxQAAAAASUVORK5CYII=",

    "bl": "iVBORw0KGgoAAAANSUhEUgAAAMUAAADFCAYAAADkODbwAAADYklEQVR4nO3dUZKUSBRA0eqJ3sNsxuW7GVcx8+0NaCsNin4F53waSlBat5NMIfn4+e+P/x4bfvz6+bH163B1/3z3CcA0ooAQBYQoIEQBIQoIUUCIAkIUEKKAEAWEKCBEASEKCFFAiALi82/+UB9M8kASV7I8Uuw9qQdX4fIJQhQQooAQBYQoIEQBIQoIUUCIAkIUEKKAEAWEKCBEASEKCFFAiAJCFBCigBAFhCggdqPY2rXDTh7cwZf7PomAO3L5BCEKCFFAiAJid6Jt02TuykgBIQoIUUCIAkIUEKKAEAWEKCBEASEKCFFAiAJCFBCigBAFhCggRAEhCghRQIgCQhQQooAQBYQoIEQBIQoIUUCIAgAAAADgNB+vfFe2V4Txjvw/BYQoIEQBIQoIUUAsv1x+a7XKKhNXYqSAEAWEKCBEAbEUxd4tIa+8VQTO5t4nCJdPEKKAEAWEKCB2b/N4PH6fKD8zITex5gp2V5/c+8RduXyCEAWEKCBEAfHl6tMWk2quzkgBIQoIUUCIAkIUEIc8ZGRFiisxUkCIAkIUEKKAEAXE8qqRh4y4OiMFhCggRAEhCohDNliGK/n0RYffuXyCEAWEKCBEAbH8Hm24OiMFhCggRAEhCghRQLz0PdorrHYxxZiRYkqcMCYKmEIUEKKAEAXEt6z42NSZyYwUEKKAEAWEKCBEAXHIFjdHrQRtHcftH5xt/Ehh6ZWzjY8CziYKCFFAjI9ib6JtAs6rLO/79Mov4+qxV36/CTvPGj9SwNlEASEKCFFAfLnFTSenf5rYPjOZXT3GUednos2zxuz79HisrXitro6Jgme5fIIQBYQoIMZH4TYPzjZq8nn2F93kmy3jR4pXMtqw5dZRwBZRQIgCQhQQh2xxs2p11eeIVSKTap41fqSwbMrZxkcBZxMFhCggRAGxvMXNCis+azxMNYORAkIUEKKAGB/FXeYld/mc72B8FHcnlvPtrj7tWVkhOcp3bKT8yteevXIj6Une4b66LUYKCFFAiAJCFFzGUfPJ5Yn2d9xGMOn92kecy+oxJn3+I0y/FWX8SDHpL3DKD4R39g6fZ3wUcDZRDPEOP0HvQhQQ46N45wnlCp9zjt3Vp0knP+nd3Wcf+6jzu8u/5xHGjxRwNlFAiAJCFBCigPgf1Kn03HMEtMEAAAAASUVORK5CYII=",

    "br": "iVBORw0KGgoAAAANSUhEUgAAAMUAAADFCAYAAADkODbwAAADYElEQVR4nO3dQXKbMABA0aSTO/QyOX4uk1O0284PeKAjbAHvLTseB7v+kaUIeH+Dm/r6/fln6d9/PftAYHaigBAFhCggRAEhCghRQIgCQhQQooAQBYQoIEQBIQoIUUCIAuLj1QcAI/XEoc/vr/e9z2Gk4NLWzq57RBQQooAQBYQoIEQBIQoIUUCIAkIUEKKAEAWEKCBEASEKCFFAiAJCFBCigBAFhCggRAEhCi5v6Yoej67ysfuaODCz/7mkTRkpIEQBIQoIUUCYaHNba5NyIwWEKCBEASEKCFFAiAJCFBCigBAFhCggRAEhCghRQIgCQhQQooAQBYQoIEQBIQoIUUCIAkIUEKKAEAWEKCBEASEKAAAAAICncXsvTmnE/bLX+DsFhCggRAEhCghRQFh94lKWVqU+v78WP+duLg8biQJCFBCigBAFl7E2cd67JcTqE6dk7xM8kSggRAEhCggTbS5hy8T73+0ejx4vCi7F3ic4gCggRAEhCggTbW7LRBs2EgWEKCBEASEKCKtPXMqIk4+MFBCigBAFhCggRAHx8eoDgKOtnWS0xkgBIQoIUUCIAkIUXJ4LLHNr9j7BAUQBIQoIUUCYaHNbLnEDG4kCQhQQooAQBYTVJ6Zx5L2x9zBSMIVZgnh7EwX8IAoIUUCIAsLqE4cZMXnee3maEYwUEKKAEAWEKCBEAeECyzzd0orSqG0eLnHD6bxiiXUvUUCIAkIUEKLgEGsT3pnOm1hj9YnNZloh2mNtcu+6T7CRKCBEASEKiOn/usg8Hk2QO5kd9dilx+89tr0/UxRstvZB2rO6s3cl6BV8fYIQBYQoIETBIc68zcNEm0WzbMV4BSMFP5zht/mRRAEhCghRQIgCwklGbDZihWjvJP4Vk34jBU8109LrGlFAiAJCFBCigLD6NMCRJ9+w7MiVMCMFhCggRAEhikncZbv2GV6nKJ7sDB+Kuxuy+vSK/+gzrNaM2Odz5CVhRr2HM4U+YmXPSAEhCghRQAyJ4gzf72GrYds8jgpjpkncXnvvFz3iPTzyHtVnOJYR7+H0X5+uNgq94vXM9B7OdCxrpo8Cnk0UkzjDb9C7EAXE9FGceaK9h9c5j48zHOSSM2wtmf1yLkf/vCOf/8jnnn6kgGcTBYQoIEQBIQqIvxuG4GwLOIU1AAAAAElFTkSuQmCC",

}





def frame_uris() -> dict:

    """Corner pieces of the Dzongkha frame as data URIs."""

    return {k: "data:image/png;base64," + v for k, v in FRAME_B64.items()}





CSS = """

<style>

@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Manrope:wght@400;500;600;700&family=Noto+Serif+Tibetan:wght@400;600&display=swap');

:root{

  /* colours taken from the ornament image */

  --cream:#fff5dc; --ink:#3a1d10; --muted:#8a6446; --red:#e5432d; --red-dark:#b82f1c;

  --orange:#f5a623; --green:#14a56b;

  --glass:rgba(255,250,236,.88); --glass-strong:rgba(255,251,240,.95);

  --line:rgba(229,67,45,.30); --user:rgba(245,166,35,.24);

  --ok:#0b7a4d; --warn:#9a5a00;

  --sbw:336px;                    /* sidebar width (Streamlit default) */

  --band-gap:calc(46vh + 1.5rem); /* space reserved for the ornament band */

}

html, body, [class*="css"], .stApp { font-family:'Manrope',system-ui,sans-serif; color:var(--ink); }



/* background image: ornament band on the left, faint pattern everywhere else */

.stApp {

  background: url("__BG__") var(--sbw) 0 / auto 100vh no-repeat fixed, var(--cream);

}

/* when the sidebar is collapsed the band moves to the left edge */

.stApp:has(section[data-testid="stSidebar"][aria-expanded="false"]),

.stApp:has([data-testid="stExpandSidebarButton"]),

.stApp:has([data-testid="stSidebarCollapsedControl"]) {

  background-position: 0 0, 0 0;

}

header[data-testid="stHeader"] { background:transparent; }

[data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stDeployButton"],

#MainMenu, footer { display:none !important; }

/* keep the "open sidebar" arrow visible so the mode menu can always be reopened */

[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"],

[data-testid="stSidebarCollapseButton"] { display:flex !important; visibility:visible !important; opacity:1 !important; }

[data-testid="stExpandSidebarButton"] svg, [data-testid="stSidebarCollapsedControl"] svg { color:var(--red-dark); }

[data-testid="stBottom"], [data-testid="stBottom"] > div { background:transparent !important; }



/* content sits to the right of the ornament band */

.block-container, [data-testid="stBottomBlockContainer"] {

    width:calc(100% - var(--band-gap) - 1.5rem) !important; max-width:680px !important;

    margin-left:var(--band-gap) !important; margin-right:auto !important; }

.block-container { padding-top:1.6rem; padding-bottom:7rem; }

[data-testid="stBottomBlockContainer"] { padding-left:4.8rem; padding-right:4.8rem; box-sizing:border-box; }



/* Dzongkha text */

.dz, .dz * { font-family:'Noto Serif Tibetan','Jomolhari','Microsoft Himalaya','Kailasa',serif !important;

             font-size:1.25rem; line-height:2.0; color:var(--ink); }



/* heading */

.hero h1 { font-family:'Sora',sans-serif; font-size:2.3rem; font-weight:700; letter-spacing:-.02em;

           margin:0 0 6px; color:var(--red-dark); }

.hero p { margin:0; color:var(--muted); font-size:1rem; }

.hero .tib { font-family:'Noto Serif Tibetan',serif; font-size:1.1rem; color:var(--red); margin-top:8px; }

.hero { margin-bottom:22px; }



/* welcome */

.welcome { padding:22px 0 8px; }

.welcome h2 { font-family:'Sora',sans-serif; font-size:1.35rem; font-weight:600; color:var(--ink); margin-bottom:4px; }

.welcome p  { color:var(--muted); margin-bottom:0; }



/* buttons + suggestion chips */

div[data-testid="stButton"], [data-testid="stElementContainer"]:has(> div[data-testid="stButton"]) { width:100% !important; }

div[data-testid="stButton"] > button { width:100%; border-radius:12px; border:1px solid var(--line);

    background:var(--glass); color:var(--ink); font-weight:600;

    padding:.65rem .9rem; transition:border-color .15s ease, box-shadow .15s ease; }

div[data-testid="stButton"] > button:hover { border-color:var(--red); color:var(--red-dark);

    box-shadow:0 2px 14px rgba(229,67,45,.22); }

div[data-testid="stButton"] > button[kind="primary"] { background:linear-gradient(135deg,var(--red),var(--orange));

    border:none; color:#fff; font-weight:700; }

div[data-testid="stButton"] > button[kind="primary"]:hover { box-shadow:0 3px 18px rgba(229,67,45,.45); color:#fff; }



/* chat bubbles */

[data-testid="stChatMessage"] { background:var(--glass); border:1px solid var(--line);

    border-radius:18px; padding:14px 18px; margin-bottom:12px; box-shadow:0 2px 10px rgba(120,60,20,.08); }

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background:var(--user); }



/* round chat avatars (images) */

[data-testid="stChatMessage"] img { border-radius:50%; object-fit:cover; }



/* badges */

.badge { display:inline-flex; align-items:center; gap:6px; font-size:.76rem; font-weight:600; padding:3px 10px;

         border-radius:999px; margin-top:6px; }

.badge.ok   { background:rgba(20,165,107,.12); color:var(--ok); border:1px solid rgba(20,165,107,.40); }

.badge.warn { background:rgba(245,166,35,.18); color:var(--warn); border:1px solid rgba(245,166,35,.55); }



/* chat input */

[data-testid="stChatInput"] { border-radius:16px; border:1px solid var(--line); background:var(--glass-strong);

    box-shadow:0 2px 16px rgba(229,67,45,.14); }

[data-testid="stChatInput"] textarea { font-family:'Manrope','Noto Serif Tibetan',sans-serif; color:var(--ink); }



/* sidebar */

section[data-testid="stSidebar"] { background:rgba(255,245,220,.96); border-right:2px solid var(--line); }

.brand { margin:4px 0 18px; position:relative; text-align:center; padding:30px 18px; min-height:92px; box-sizing:border-box;

         border-radius:6px; background-color:rgba(255,250,236,.88);

         background-image:url("__TL__"), url("__TR__"), url("__BL__"), url("__BR__");

         background-position:top left, top right, bottom left, bottom right;

         background-size:56px 56px; background-repeat:no-repeat; }

.brand .t { font-family:'Sora',sans-serif; font-weight:700; font-size:1.25rem; color:var(--red-dark); line-height:1.15; }

.brand .s { font-size:.8rem; color:var(--red); margin-top:3px; }

.side-label { font-size:.78rem; color:var(--muted); margin:14px 0 4px; font-weight:600; }





/* Actual Streamlit Material-icon navigation buttons */

section[data-testid="stSidebar"] div[class*="st-key-nav_"] button {

    width:100% !important; max-width:none !important;

    min-height:54px; justify-content:flex-start !important;

    border-radius:13px !important; padding:10px 16px !important;

    font-size:0.96rem !important; font-weight:650 !important;

    margin-bottom:7px !important; box-shadow:0 2px 9px rgba(87,42,25,.05);

    transition:background .15s,transform .15s,border-color .15s;

}

section[data-testid="stSidebar"] div[class*="st-key-nav_"] button:hover { transform:translateX(3px); }

section[data-testid="stSidebar"] div[class*="st-key-nav_"] button span[data-testid="stIconMaterial"] {

    font-size:1.35rem !important; margin-right:10px !important;

}

section[data-testid="stSidebar"] div[class*="st-key-nav_"] button[kind="primary"] {

    background:#b73527 !important; color:white !important;

    border-color:#b73527 !important;

}



/* Premium, accessible sidebar mode cards — native Streamlit radio stays interactive */

section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] {

    display:flex; flex-direction:column; gap:10px; margin:10px 0 18px;

}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] {

    display:flex !important; align-items:center; min-height:62px;

    padding:12px 14px !important; border:1px solid rgba(151,84,58,.20);

    border-radius:16px; background:rgba(255,255,255,.66);

    box-shadow:0 2px 8px rgba(76,43,24,.035);

    transition:background .18s ease, border-color .18s ease, box-shadow .18s ease, transform .18s ease;

    cursor:pointer;

}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {

    background:#fffaf1; border-color:rgba(193,65,42,.45);

    box-shadow:0 6px 18px rgba(107,54,35,.09); transform:translateY(-1px);

}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {

    background:linear-gradient(110deg,#fff8ee,#ffece1);

    border-color:#d65c3f; box-shadow:inset 3px 0 0 #c83c28, 0 5px 16px rgba(172,65,37,.12);

}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]:focus-within {

    outline:2px solid #ba5035; outline-offset:2px;

}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {

    margin-right:8px;

}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] p {

    font-family:'Manrope',sans-serif; font-size:.95rem; font-weight:650;

    letter-spacing:.005em; color:#3c271f; line-height:1.35;

}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {

    color:#a42c1c;

}

.mode-caption {font-size:.79rem;color:#866e60;line-height:1.5;margin:2px 0 8px;}

/* expanders */

[data-testid="stExpander"] { background:var(--glass); border:1px solid var(--line); border-radius:12px; }



/* tool pages (summarize / next word) */

.tool-head { text-align:center; margin:6px auto 26px; max-width:760px; }

.tool-head h1 { font-family:'Sora',sans-serif; font-size:2.5rem; font-weight:700; letter-spacing:-.02em;

                color:var(--red-dark); margin:0 0 10px; }

.tool-head p { color:var(--muted); font-size:1.05rem; margin:0; }

.panel-head { display:flex; justify-content:space-between; align-items:baseline; margin:0 2px 8px; }

.panel-head .name { font-family:'Sora',sans-serif; font-weight:600; font-size:1rem; color:var(--ink); }

.panel-head .count { font-size:.82rem; color:var(--muted); }

[data-testid="stTextArea"] textarea { background:var(--glass-strong) !important;

    border:1px solid var(--line) !important; border-radius:16px !important; color:var(--ink) !important;

    font-size:1rem; line-height:1.6; padding:16px !important; }

[data-testid="stTextArea"] textarea:focus { border-color:var(--red) !important; box-shadow:0 0 0 1px var(--red) !important; }

.out-panel { background:var(--glass-strong); border:1px solid var(--line);

    border-radius:16px; padding:16px 18px; height:340px; overflow-y:auto; }

.out-panel.empty { display:flex; align-items:center; justify-content:center; text-align:center;

    color:var(--muted); }

.steps { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:34px 0 8px; }

.steps div { background:var(--glass); border:1px solid var(--line); border-radius:14px; padding:14px 16px;

    color:var(--muted); font-size:.92rem; line-height:1.5; }

.steps b { display:block; color:var(--red-dark); font-family:'Sora',sans-serif; margin-bottom:2px; }

@media (max-width:760px){ .steps{ grid-template-columns:1fr; } .tool-head h1{ font-size:1.9rem; } }



/* result cards (next word) */

.card { background:var(--glass); border:1px solid var(--line); border-radius:16px;

    padding:16px 20px; margin:8px 0 12px; }

.card h4 { margin:0 0 8px; font-size:.9rem; color:var(--muted); font-weight:600; }

.pred { display:flex; align-items:center; gap:12px; padding:8px 0; border-bottom:1px dashed var(--line); }

.pred:last-child { border-bottom:none; }

.pred .rk { width:26px; height:26px; border-radius:50%; background:var(--red); color:#fff; font-size:.78rem;

            display:flex; align-items:center; justify-content:center; font-weight:700; }

.pred .bar { flex:1; height:8px; background:rgba(58,29,16,.10); border-radius:99px; overflow:hidden; }

.pred .bar > i { display:block; height:100%; background:linear-gradient(90deg,var(--orange),var(--red)); }

.pred .pc { color:var(--muted); font-size:.82rem; width:52px; text-align:right; }

.foot { text-align:center; color:var(--muted); font-size:.78rem; margin-top:22px; }



/* phones / small windows: no room beside the band, so keep it as a soft backdrop */

@media (max-width:900px){

  .stApp, .stApp:has(section[data-testid="stSidebar"][aria-expanded="false"]) {

    background: linear-gradient(rgba(255,245,220,.86), rgba(255,245,220,.86)),

                url("__BG__") -20vh 0 / auto 100vh no-repeat fixed, var(--cream); }

  .block-container, [data-testid="stBottomBlockContainer"] {

    width:100% !important; margin-left:auto !important; max-width:100% !important;

    padding-left:1rem; padding-right:1rem; }

  [data-testid="stChatMessage"], .card, .out-panel, .steps div { background:rgba(255,251,240,.96); }

}



@media (prefers-reduced-motion: reduce){ * { transition:none !important; } }

</style>

""".replace("__BG__", bg_data_uri())

_f = frame_uris()

for _k, _v in _f.items():

    CSS = CSS.replace(f"__{_k.upper()}__", _v)

st.markdown(CSS, unsafe_allow_html=True)





# ----------------------------------------------------------------- UI helpers

def dz_html(text: str) -> str:

    return f'<div class="dz">{_html.escape(text)}</div>'





def syllable_count(text: str) -> int:

    return len([s for s in re.split(r"[་།\s]+", text) if s])





def copy_button(text: str):

    """Small copy-to-clipboard button (runs in its own iframe, so it carries its own style)."""

    payload = json.dumps(text)

    components.html(f"""

    <button id="c" style="font:600 14px Manrope,system-ui,sans-serif;width:100%;padding:10px 14px;border-radius:12px;

      border:1px solid rgba(229,67,45,.35);background:#fff8e6;color:#3a1d10;cursor:pointer">Copy summary</button>

    <script>

      const b=document.getElementById('c'), t={payload};

      b.onclick=async()=>{{

        try{{ await navigator.clipboard.writeText(t); }}

        catch(e){{ const a=document.createElement('textarea'); a.value=t; document.body.appendChild(a); a.select();

                  document.execCommand('copy'); a.remove(); }}

        b.textContent='Copied'; setTimeout(()=>b.textContent='Copy summary',1500);

      }};

    </script>""", height=48)





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

    # Dzoseg is an optional public Dzongkha tokenizer. The helper falls back

    # to traditional Tseg segmentation if the package is not installed.

    tokens = segment_dzongkha(rep.fixed, method="tseg")

    if tokens:

        notes.append(f"Dzongkha syllable segmentation: {len(tokens)} units")

    return rep.fixed, notes





def render_assistant(content, source=None, notes=None, debug=None):

    st.markdown(dz_html(content) if TIBETAN.search(content) else content, unsafe_allow_html=True)

    if source == "verified":

        st.markdown('<span class="badge ok">✔ Verified answer</span>', unsafe_allow_html=True)

    elif source == "web":

        st.markdown('<span class="badge ok">🌐 Answer from web search · translated automatically</span>',

                    unsafe_allow_html=True)

    elif source == "general":

        st.markdown('<span class="badge warn">ℹ General AI answer · translated automatically, facts may be wrong</span>',

                    unsafe_allow_html=True)

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

        with st.chat_message(m["role"], avatar=USER_AVATAR if m["role"] == "user" else BOT_AVATAR):

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

        if cols[i % 2].button(label, key=f"sg{i}"):

            st.session_state["pending"] = prompt

            st.rerun()





def chat_header():

    st.markdown(

        f'<div class="hero"><h1>{APP_NAME}</h1>'

        '<p>Ask in English or Dzongkha. Answers come back in Dzongkha.</p>'

        '<div class="tib">འབྲུག་རྒྱལ་ཁབ་ཀྱི་རྒྱལ་ཡོངས་སྐད་ཡིག</div></div>', unsafe_allow_html=True)





def tool_header(title, subtitle):

    st.markdown(f'<div class="tool-head"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)





def widen_page():

    """Tool pages use a wider column, but still start to the right of the ornament band."""

    st.markdown("<style>.block-container{max-width:1000px !important;}</style>", unsafe_allow_html=True)





def clear_summary():

    st.session_state["sum_text"] = ""

    st.session_state.pop("sum_out", None)





# ------------------------------------------------------------ chat history

# Saved to chat_sessions.json next to this file (separate from any old chat_history.json).

HISTORY_FILE = Path(__file__).resolve().parent / "chat_sessions.json"





def load_sessions():

    try:

        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))

        return data if isinstance(data, list) else []

    except Exception:

        return []





def write_sessions(sessions):

    try:

        HISTORY_FILE.write_text(json.dumps(sessions, ensure_ascii=False, indent=1), encoding="utf-8")

    except Exception:

        pass





def save_current(kind):

    """Save the chat in the current mode ('pipe' or 'ft') to the history file."""

    msgs = st.session_state["hist_pipe" if kind == "pipe" else "hist_ft"]

    if not msgs:

        return

    idk = f"chat_id_{kind}"

    sid = st.session_state.get(idk) or uuid.uuid4().hex[:10]

    st.session_state[idk] = sid

    first = next((m["content"] for m in msgs if m["role"] == "user"), "Chat")

    rec = {"id": sid, "kind": kind, "title": first[:40], "ts": time.time(), "messages": msgs,

           "en_hist": st.session_state.en_hist if kind == "pipe" else []}

    sessions = [x for x in load_sessions() if x.get("id") != sid]

    sessions.insert(0, rec)

    write_sessions(sessions[:50])





def open_session(sid):

    for x in load_sessions():

        if x.get("id") == sid:

            if x["kind"] == "pipe":

                st.session_state.hist_pipe = x["messages"]

                st.session_state.en_hist = x.get("en_hist", [])

                st.session_state["mode_radio"] = "💬  Chat"

            else:

                st.session_state.hist_ft = x["messages"]

                st.session_state["mode_radio"] = "🎓  Chat (fine-tuned)"

            st.session_state[f"chat_id_{x['kind']}"] = sid

            return





def delete_session(sid):

    write_sessions([x for x in load_sessions() if x.get("id") != sid])

    for k in ("chat_id_pipe", "chat_id_ft"):

        if st.session_state.get(k) == sid:

            st.session_state.pop(k)





# -------------------------------------------------------------------- state

for key, default in [("hist_pipe", []), ("en_hist", []), ("hist_ft", []), ("en_hist_ft", [])]:

    st.session_state.setdefault(key, default)

st.session_state.setdefault("sum_text", "")



# ------------------------------------------------------------ welcome splash

# Shown once when the page opens. Needs welcome_bg.jpg next to this file (or in ./assets).

def _find_welcome_bg():

    here = Path(__file__).resolve().parent

    for p in (here / "welcome_bg.jpg", here / "assets" / "welcome_bg.jpg",

              Path.cwd() / "welcome_bg.jpg", Path.cwd() / "assets" / "welcome_bg.jpg"):

        if p.exists():

            return p

    return None





SPLASH_CSS = """

<style>

.stApp, .stApp:has([data-testid="stExpandSidebarButton"]), .stApp:has(section[data-testid="stSidebar"]) {

  background:

    linear-gradient(to top, rgba(255,245,220,.94) 0%, rgba(255,245,220,.80) 24%, rgba(255,245,220,0) 52%) fixed,

    url("__IMG__") center / cover no-repeat fixed !important;

}

.stApp section[data-testid="stSidebar"], .stApp [data-testid="stExpandSidebarButton"],

.stApp [data-testid="stSidebarCollapsedControl"], .stApp header[data-testid="stHeader"] { display:none !important; }

.stApp .block-container, .stApp [data-testid="stMainBlockContainer"] {

  width:100% !important; max-width:760px !important; margin:0 auto !important;

  min-height:100vh; display:flex; flex-direction:column; justify-content:flex-end;

  padding:0 1.2rem 7vh !important; text-align:center; }

.splash { animation:spin .9s ease both; }

.splash .tib { font-family:'Noto Serif Tibetan',serif; font-size:1.15rem; color:var(--red); margin-bottom:6px; }

.splash h1 { font-family:'Sora',sans-serif; font-weight:700; letter-spacing:-.02em; line-height:1.15;

             font-size:clamp(1.8rem,4.6vw,3.1rem); color:var(--ink); margin:0 0 8px; }

.splash h1 span { color:var(--red-dark); }

.splash p { color:var(--muted); font-size:1.02rem; margin:0 0 20px; }

.stApp [data-testid="stButton"] { max-width:280px; margin:0 auto !important; }

@keyframes spin { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:none; } }

@media (prefers-reduced-motion: reduce){ .splash { animation:none; } }

</style>

"""



if not st.session_state.get("entered"):

    _bg = _find_welcome_bg()

    if _bg is None:

        st.warning("welcome_bg.jpg not found. Put it next to the app file (or in an 'assets' folder).")

        _uri = ""

    else:

        _uri = "data:image/jpeg;base64," + _read_b64(str(_bg), _bg.stat().st_mtime)

    st.markdown(SPLASH_CSS.replace("__IMG__", _uri), unsafe_allow_html=True)

    st.markdown(

        '<div class="splash"><div class="tib">འབྲུག་རྒྱལ་ཁབ་ཀྱི་རྒྱལ་ཡོངས་སྐད་ཡིག</div>'

        '<h1>Welcome to <span>Dzongkha AI Assistant</span></h1>'

        '<p>Ask in English or Dzongkha. Answers come back in Dzongkha.</p></div>',

        unsafe_allow_html=True)

    if st.button("Get started  →", type="primary", key="enter"):

        st.session_state["entered"] = True

        st.rerun()

    st.stop()



# ------------------------------------------------------------------ sidebar

with st.sidebar:

    st.markdown('<div class="brand"><div class="t">Dzongkha AI</div>'

                '<div class="s">English ⇄ རྫོང་ཁ</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="side-label">WORKSPACE</div><div class="mode-caption">Choose a tool to get started</div>', unsafe_allow_html=True)

    # Real icon buttons instead of the old emoji radio menu.

    # Material icons are rendered by Streamlit and each mode has its own icon.

    NAV_ITEMS = [

        ("Chat", "Chat (translate pipeline)", ":material/chat_bubble_outline:", "Ask questions"),

        ("Fine-tuned Chat", "Chat (fine-tuned)", ":material/school:", "Specialised answers"),

        ("Summarize", "Summarize", ":material/article:", "Summarise text"),

        ("Next Word", "Next word", ":material/auto_awesome:", "Word suggestions"),

        ("Translate", "Translate", ":material/translate:", "English and Dzongkha"),

    ]

    if "active_nav" not in st.session_state:

        st.session_state.active_nav = "Chat (translate pipeline)"

    for title, nav_value, nav_icon, subtitle in NAV_ITEMS:

        active = st.session_state.active_nav == nav_value

        if st.button(

            title,

            icon=nav_icon,

            key="nav_" + nav_value.lower().replace(" ", "_").replace("(", "").replace(")", ""),

            type="primary" if active else "secondary",

            use_container_width=True,

            help=subtitle,

        ):

            st.session_state.active_nav = nav_value

            st.rerun()

    mode = st.session_state.active_nav



    if st.button("🗑️  New chat"):

        st.session_state.hist_pipe, st.session_state.en_hist, st.session_state.hist_ft = [], [], []

        st.session_state.en_hist_ft = []

        st.session_state.pop("pending", None)

        st.session_state.pop("chat_id_pipe", None)

        st.session_state.pop("chat_id_ft", None)

        st.rerun()



    hist_slot = st.container()   # filled at the bottom of the script so it is always up to date



    st.markdown('<div class="side-label">About</div>', unsafe_allow_html=True)

    st.caption("✔ Verified = answer from the hand-checked answer bank.\n\n"

               "⚠ Machine translated = cleaned and spell-checked automatically, but not guaranteed.")



# --------------------------------------------------------------------- modes

if mode == "Chat (translate pipeline)":

    chat_header()

    if not st.session_state.hist_pipe:

        welcome()

    render_history("hist_pipe")

    if q := get_prompt("Message the assistant (English or Dzongkha)…"):

        st.session_state.hist_pipe.append({"role": "user", "content": q})

        with st.chat_message("user", avatar=USER_AVATAR):

            st.write(q)

        with st.chat_message("assistant", avatar=BOT_AVATAR):

            with st.spinner("Thinking…"):

                try:

                    dz, en_q, en_a, source = pipeline_chat(st.session_state.en_hist, q, DZ_REPO, MAX_TOKENS, TEMPERATURE)

                except Exception as e:

                    st.error(f"Something went wrong: {e}")

                    st.stop()

            notes = []

            if source != "verified":

                dz, notes = check_notes(dz, load_bank_and_lexicon()[1])

            render_assistant(dz, source, notes, (en_q, en_a))

            if TIMES:

                st.caption("⏱ " + " · ".join(f"{k} {v:.1f}s" for k, v in TIMES.items())

                           + (" · engine: Ollama" if ollama_ready() else " · engine: MLX"))

        st.session_state.hist_pipe.append({"role": "assistant", "content": dz, "source": source,

                                           "notes": notes, "debug": (en_q, en_a)})

        save_current("pipe")



elif mode == "Chat (fine-tuned)":

    chat_header()

    if not st.session_state.hist_ft:

        welcome()

    render_history("hist_ft")

    if q := get_prompt("Message the assistant (English or Dzongkha)…"):

        st.session_state.hist_ft.append({"role": "user", "content": q})

        with st.chat_message("user", avatar=USER_AVATAR):

            st.write(q)

        with st.chat_message("assistant", avatar=BOT_AVATAR):

            with st.spinner("Thinking…"):

                try:

                    # Questions covered by your training data -> your fine-tuned LoRA model.

                    # Everything else -> the full pipeline (verified answers, facts, live web, general AI).

                    if in_training_data(q) and Path(ADAPTER_PATH).exists():

                        reply, ok = dzongkha_only(SYSTEM, [{"role": "user", "content": q}], MAX_TOKENS, TEMPERATURE)

                        source, debug = "machine", None

                        if not ok:

                            raise RuntimeError("LoRA reply was not clean Dzongkha")

                    else:

                        raise RuntimeError("not in training data")

                except Exception:

                    TIMES.clear()

                    reply, en_q, en_a, source = pipeline_chat(st.session_state.en_hist_ft, q, DZ_REPO, MAX_TOKENS, TEMPERATURE)

                    debug = (en_q, en_a)

            notes = []

            if source != "verified":

                reply, notes = check_notes(reply, load_bank_and_lexicon()[1])

            render_assistant(reply, source, notes, debug)

            if TIMES:

                st.caption("⏱ " + " · ".join(f"{k} {v:.1f}s" for k, v in TIMES.items()))

        st.session_state.hist_ft.append({"role": "assistant", "content": reply, "source": source,

                                         "notes": notes, "debug": debug})

        save_current("ft")



elif mode == "Translate":

    widen_page()

    tool_header("Translate", "Translate words, sentences or paragraphs directly — without asking the chat model.")

    direction = st.radio("Translation direction", ["English → Dzongkha", "Dzongkha → English"], horizontal=True)

    source_text = st.text_area("Text to translate", height=170,

                               placeholder="Type or paste the exact words or sentences to translate…")

    if st.button("🌐 Translate now", type="primary"):

        if not source_text.strip():

            st.warning("Please enter text to translate.")

        else:

            with st.spinner("Translating directly (first request may take longer)…"):

                try:

                    if direction == "English → Dzongkha":

                        result = normalize(translate(source_text.strip(), DZ_REPO, EN, DZ, max_new=128))

                    else:

                        result = translate(source_text.strip(), DZ_TO_EN, DZ, EN, max_new=128)

                    st.session_state["direct_translation"] = {"direction": direction, "input": source_text, "output": result}

                except Exception as exc:

                    st.error(f"Translation failed: {exc}")

    result_data = st.session_state.get("direct_translation")

    if result_data:

        st.markdown("**Translation result**")

        st.text_area("Translated text", value=result_data["output"], height=170,

                     help="Select the text and copy it using Command+C.")

        st.caption("Machine translation can contain spelling or meaning errors. Verify important text with a fluent speaker.")



elif mode == "Summarize":

    widen_page()

    tool_header("Summarize text",

                "Paste English or Dzongkha text and get a short summary in Dzongkha.")



    left, right = st.columns(2, gap="large")



    with left:

        words = len(st.session_state["sum_text"].split())

        st.markdown(f'<div class="panel-head"><span class="name">Your text</span>'

                    f'<span class="count">{words} words</span></div>', unsafe_allow_html=True)

        text = st.text_area("Your text", key="sum_text", height=340, label_visibility="collapsed",

                            placeholder="Paste or type your text here, then press Ctrl+Enter.")

        b1, b2 = st.columns([3, 1])

        run = b1.button("Summarize", type="primary")

        b2.button("Clear", on_click=clear_summary)



    if run and text.strip():

        with st.spinner("Summarizing…"):

            try:

                dz, en_s = pipeline_summary(text, DZ_REPO, MAX_TOKENS, TEMPERATURE)

            except Exception as e:

                st.error(f"Something went wrong: {e}")

                st.stop()

        dz, notes = check_notes(dz, load_bank_and_lexicon()[1])

        st.session_state["sum_out"] = {"dz": dz, "en": en_s, "notes": notes}

    elif run:

        st.warning("Add some text first.")



    out = st.session_state.get("sum_out")

    with right:

        count = f"{syllable_count(out['dz'])} syllables" if out else ""

        st.markdown(f'<div class="panel-head"><span class="name">Summary in Dzongkha</span>'

                    f'<span class="count">{count}</span></div>', unsafe_allow_html=True)

        if out:

            st.markdown(f'<div class="out-panel">{dz_html(out["dz"])}</div>', unsafe_allow_html=True)

            copy_button(out["dz"])

            if out["notes"]:

                with st.expander("Spelling & sentence checks"):

                    for n in out["notes"]:

                        st.write("• " + n)

            with st.expander("English summary"):

                st.write(out["en"])

        else:

            st.markdown('<div class="out-panel empty">Your summary will appear here.</div>',

                        unsafe_allow_html=True)



    st.markdown(

        '<div class="steps">'

        '<div><b>Paste your text</b>English or Dzongkha, any length that fits the box.</div>'

        '<div><b>Press Summarize</b>The text is summarized, translated, and spell-checked.</div>'

        '<div><b>Copy the result</b>Review the checks, then copy the Dzongkha summary.</div>'

        '</div>', unsafe_allow_html=True)



else:

    widen_page()

    tool_header("Next-word prediction", "Type the start of a Dzongkha sentence and see likely next words.")

    _, mid, _ = st.columns([1, 2, 1])

    with mid:

        prefix = st.text_input("Dzongkha text", value="འབྲུག་རྒྱལ་ཁབ་")

        if st.button("Predict", type="primary") and prefix.strip():

            with st.spinner("Predicting…"):

                preds = predict_next_words(prefix)

            rows = "".join(

                f'<div class="pred"><div class="rk">{i}</div><div class="dz" style="min-width:90px">{_html.escape(w)}</div>'

                f'<div class="bar"><i style="width:{min(100, p * 100 * 2):.0f}%"></i></div><div class="pc">{p:.1%}</div></div>'

                for i, (w, p) in enumerate(preds, 1))

            st.markdown(f'<div class="card"><h4>Top suggestions</h4>{rows}</div>', unsafe_allow_html=True)



with hist_slot:

    st.markdown('<div class="side-label">History</div>', unsafe_allow_html=True)

    _sessions = load_sessions()

    if not _sessions:

        st.caption("Your chats will be saved here.")

    for _s in _sessions[:12]:

        c1, c2 = st.columns([5, 1])

        c1.button(f"{'💬' if _s['kind'] == 'pipe' else '🎓'} {_s['title']}", key=f"h_{_s['id']}",

                  on_click=open_session, args=(_s["id"],))

        c2.button("✕", key=f"d_{_s['id']}", on_click=delete_session, args=(_s["id"],))



st.markdown('<div class="foot">Dzongkha AI Assistant · NLP module project</div>', unsafe_allow_html=True)