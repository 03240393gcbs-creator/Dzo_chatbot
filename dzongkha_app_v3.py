"""Dzongkha AI Assistant — complete Streamlit frontend and backend.
Run: streamlit run dzongkha_app_v3_fixed.py
Place chatbot_bg.png and welcome_bg.jpg beside this file (or in assets/).
Optional project modules: dzongkha_nlp.py, govtech_dnlp.py.
"""
import base64
import html
import json
import os
import re
import time
import uuid
import urllib.request
from functools import lru_cache
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
APP_NAME = 'Dzongkha AI Assistant'
BASE_MODEL = 'mlx-community/Qwen2.5-3B-Instruct-4bit'
OLLAMA_URL = 'http://localhost:11434'
OLLAMA_MODEL = 'qwen2.5:3b'
ADAPTER_PATH = ROOT / 'data' / 'adapters'
DZ_TO_EN = 'KarmaCST/nllb-200-distilled-600M-dz-to-en'
EN_TO_DZ = 'KarmaCST/nllb-200-distilled-600M-73K-en-to-dz'
NEXTWORD_REPO = 'KarmaCST/dzongkha_nextword_gpt2'
DZ, EN = 'dzo_Tibt', 'eng_Latn'
MAX_TOKENS = 80
TEMPERATURE = 0.1
TIBETAN = re.compile(r'[\u0F00-\u0FFF]')
LETTERS = re.compile(r'[A-Za-z\u4e00-\u9fff\u0F00-\u0FFF]')
TIMES = {}

st.set_page_config(page_title=APP_NAME, page_icon='🐉', layout='wide', initial_sidebar_state='expanded')


def asset(name):
    for p in (ROOT / name, ASSETS / name):
        if p.is_file():
            return p
    return None


@st.cache_data(show_spinner=False)
def data_uri(path, mtime):
    suffix = Path(path).suffix.lower()
    mime = 'image/jpeg' if suffix in ('.jpg', '.jpeg') else 'image/png'
    return f'data:{mime};base64,' + base64.b64encode(Path(path).read_bytes()).decode('ascii')


def image_uri(name):
    p = asset(name)
    return data_uri(str(p), p.stat().st_mtime) if p else ''


def avatar(name, fallback):
    for ext in ('png', 'webp', 'jpg', 'jpeg'):
        p = asset(f'{name}.{ext}')
        if p:
            return str(p)
    return fallback


USER_AVATAR = avatar('user_avatar', '🧑')
BOT_AVATAR = avatar('bot_avatar', '🐉')

# Safe optional imports: the UI still starts when local NLP modules are absent.
try:
    from dzongkha_nlp import AnswerBank, FactBank, Lexicon, analyse, normalize
    HAS_NLP = True
except ImportError:
    HAS_NLP = False
    def normalize(s):
        return s.strip()

try:
    from govtech_dnlp import segment_dzongkha
except ImportError:
    def segment_dzongkha(text, method='tseg'):
        return [s for s in re.split(r'[་།\s]+', text) if s]


def dz_ratio(text):
    letters = LETTERS.findall(text)
    return len(TIBETAN.findall(text)) / len(letters) if letters else 0


def clean_en(text):
    return re.sub(r'\s+', ' ', re.sub(r'[*#`_>]+', ' ', text)).strip()


@st.cache_resource(show_spinner='Loading local MLX model…')
def load_base():
    from mlx_lm import load
    return load(BASE_MODEL)


@st.cache_resource(show_spinner='Loading fine-tuned model…')
def load_finetuned():
    from mlx_lm import load
    return load(BASE_MODEL, adapter_path=str(ADAPTER_PATH))


@lru_cache(maxsize=1)
def ollama_ready():
    try:
        with urllib.request.urlopen(OLLAMA_URL + '/api/tags', timeout=2) as response:
            models = json.load(response).get('models', [])
        return any(m.get('name', '').split(':')[0] == OLLAMA_MODEL.split(':')[0] for m in models)
    except Exception:
        return False


def run_ollama(system, messages, max_tokens, temp):
    payload = json.dumps({
        'model': OLLAMA_MODEL, 'stream': False,
        'messages': [{'role': 'system', 'content': system}] + messages,
        'options': {'temperature': temp, 'num_predict': max_tokens}
    }).encode('utf-8')
    req = urllib.request.Request(OLLAMA_URL + '/api/chat', data=payload,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)['message']['content'].strip()


def run_llm(system, messages, max_tokens=80, temp=0.1, finetuned=False):
    start = time.time()
    try:
        if not finetuned and ollama_ready():
            return run_ollama(system, messages, max_tokens, temp)
        from mlx_lm import generate
        model, tokenizer = load_finetuned() if finetuned else load_base()
        prompt = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': system}] + messages,
            tokenize=False, add_generation_prompt=True)
        kwargs = {}
        try:
            from mlx_lm.sample_utils import make_sampler
            kwargs['sampler'] = make_sampler(temp=temp)
        except ImportError:
            pass
        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                        verbose=False, **kwargs).strip()
    finally:
        TIMES['model'] = TIMES.get('model', 0) + time.time() - start


@st.cache_resource(show_spinner='Loading translation model…')
def load_translator(repo, src_lang):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    requested = os.getenv('DZ_TRANSLATION_DEVICE', 'auto').lower()
    available_mps = torch.backends.mps.is_available()
    if requested == 'mps' and not available_mps:
        raise RuntimeError('MPS is unavailable; set DZ_TRANSLATION_DEVICE=cpu.')
    device = 'cpu' if requested == 'cpu' else ('mps' if available_mps else 'cpu')
    tok = AutoTokenizer.from_pretrained(repo, src_lang=src_lang)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(repo, low_cpu_mem_usage=False)
    mdl.eval()
    try:
        mdl.to(device)
    except (RuntimeError, NotImplementedError):
        if requested == 'mps':
            raise
        device = 'cpu'
        mdl.to(device)
    return tok, mdl, device


CACHE_FILE = ROOT / 'translation_cache.json'


@st.cache_resource

def translation_cache():
    try:
        data = json.loads(CACHE_FILE.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def translate(text, repo, src, tgt, max_new=96):
    import torch
    start = time.time()
    key = json.dumps([repo, src, tgt, text, max_new], ensure_ascii=False)
    cache = translation_cache()
    if key in cache:
        return cache[key]
    try:
        tok, mdl, device = load_translator(repo, src)
        inputs = tok(text, return_tensors='pt', truncation=True, max_length=256).to(device)
        with torch.inference_mode():
            output = mdl.generate(**inputs, forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
                                  max_new_tokens=max_new, num_beams=1, do_sample=False,
                                  no_repeat_ngram_size=3)
        result = tok.batch_decode(output.to('cpu'), skip_special_tokens=True)[0].strip()
        if result:
            cache[key] = result
            try:
                CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
            except OSError:
                pass
        return result
    finally:
        TIMES['translate'] = TIMES.get('translate', 0) + time.time() - start


def en_to_dz(text):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_en(text)) if s.strip()][:3]
    return normalize(' '.join(translate(s, EN_TO_DZ, EN, DZ) for s in sentences))


@st.cache_resource

def load_bank():
    if not HAS_NLP:
        return None, None
    try:
        bank = AnswerBank(str(ROOT / 'verified_kb.json'))
        lex = Lexicon()
        bank.build_lexicon(lex)
        for filename in ('data/lexicon.txt', 'data/extra_corpus.txt'):
            p = ROOT / filename
            if p.is_file():
                lex.add_file(str(p))
        return bank, lex
    except Exception:
        return None, None


@st.cache_resource

def load_facts():
    if not HAS_NLP or not (ROOT / 'facts_en.json').is_file():
        return None
    try:
        return FactBank(str(ROOT / 'facts_en.json'))
    except Exception:
        return None


@st.cache_data

def training_questions():
    questions = []
    for filename in ('data/dataset.json', 'dzo_dataset_full.json'):
        try:
            data = json.loads((ROOT / filename).read_text(encoding='utf-8'))
            questions += [x['instruction'] for x in data if isinstance(x, dict) and x.get('instruction')]
        except (OSError, ValueError, TypeError):
            pass
    return questions


def in_training_data(question, threshold=0.6):
    def tokens(s):
        return set(re.findall(r'[\w\u0F00-\u0FFF]+', s.lower()))
    qt = tokens(question)
    return bool(qt) and any(len(qt & (kt := tokens(q))) / len(qt | kt) >= threshold
                            for q in training_questions() if tokens(q))


def extract_translation_request(message):
    text = message.strip()
    if not (re.search(r'\b(dzongkha|dzongka|dzo)\b', text, re.I)
            and re.search(r'\b(translate|translation|say|write|convert)\b', text, re.I)):
        return None
    quoted = re.search(r'[“"]([^”"]{1,600})[”"]|[‘\']([^’\']{1,600})[’\']', text)
    if quoted:
        return (quoted.group(1) or quoted.group(2)).strip()
    for pattern in (
        r'\b(?:translate|convert|write|say)\s+(.+?)\s+(?:into|to|in)\s+(?:dzongkha|dzongka|dzo)\b',
        r'\b(?:dzongkha|dzongka|dzo)\s*:\s*(.+)$',
        r'\b(?:translate|convert|write|say)\b.*?\b(?:dzongkha|dzongka|dzo)\s*:\s*(.+)$',
    ):
        match = re.search(pattern, text, re.I | re.S)
        if match:
            result = match.group(1).strip(' .?!:;\n\t')
            if result and result.lower() not in ('this', 'it', 'that'):
                return result
    return None


EN_SYSTEM = (
    'You are a helpful assistant. Here dzo means the Dzongkha language. '
    'Answer in at most three short, factual English sentences. '
    'Do not guess unknown facts. No lists, markdown or emojis.'
)


def pipeline_chat(history, question):
    TIMES.clear()
    phrase = extract_translation_request(question)
    if phrase is not None:
        if dz_ratio(phrase) >= 0.5:
            return normalize(phrase), phrase, 'Already Dzongkha', 'machine'
        return normalize(translate(phrase, EN_TO_DZ, EN, DZ)), phrase, 'Direct translation', 'machine'

    bank, _ = load_bank()
    if bank:
        try:
            item, score = bank.lookup(question)
            if item:
                return item['dz_answer'], question, 'Verified answer bank', 'verified'
        except Exception:
            pass

    en_q = translate(question, DZ_TO_EN, DZ, EN) if dz_ratio(question) >= 0.5 else question
    facts_db = load_facts()
    facts = facts_db.lookup(en_q) if facts_db else None
    if facts:
        en_a = ' '.join(f['answer_en'] for f in facts)
        source = 'machine'
    else:
        messages = history[-8:] + [{'role': 'user', 'content': en_q}]
        en_a = clean_en(run_llm(EN_SYSTEM + '\nToday is ' + time.strftime('%d %B %Y') + '.',
                               messages, min(MAX_TOKENS, 120), TEMPERATURE))
        history.extend([{'role': 'user', 'content': en_q},
                        {'role': 'assistant', 'content': en_a}])
        source = 'general'
    return en_to_dz(en_a), en_q, en_a, source


def check_notes(text):
    _, lex = load_bank()
    if lex is None:
        return normalize(text), []
    try:
        rep = analyse(text, lex)
        notes = []
        if rep.changes:
            notes.append('Auto-fixed: ' + ', '.join(f'{a} → {b}' for a, b in rep.changes))
        if rep.malformed:
            notes.append('Malformed syllables: ' + ', '.join(rep.malformed))
        if rep.unknown:
            notes.append('Not in word list: ' + ', '.join(w for w, _ in rep.unknown[:8]))
        notes += list(rep.structure)
        notes.append(f'Dzongkha segmentation: {len(segment_dzongkha(rep.fixed, method="tseg"))} units')
        return rep.fixed, notes
    except Exception as exc:
        return text, [f'Spelling checker unavailable: {exc}']


@st.cache_resource(show_spinner='Loading Dzongkha next-word model…')
def load_nextword():
    import torch
    from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tok = PreTrainedTokenizerFast.from_pretrained(NEXTWORD_REPO)
    mdl = GPT2LMHeadModel.from_pretrained(NEXTWORD_REPO).to(device).eval()
    return mdl, tok, device


def predict_next_words(text, k=5):
    import torch
    model, tokenizer, device = load_nextword()
    ids = tokenizer.encode(text, return_tensors='pt').to(device)
    with torch.no_grad():
        probs = torch.softmax(model(ids).logits[:, -1, :], dim=-1)
        top_p, top_i = torch.topk(probs, 50)
    result = []
    for i, p in zip(top_i[0], top_p[0]):
        word = tokenizer.decode([int(i)]).strip()
        if word and not re.search(r'<0x[0-9A-Fa-f]+>', word):
            result.append((word, p.item()))
        if len(result) >= k:
            break
    return result


# -------------------------- VISUAL DESIGN --------------------------
# Background is the whole cream textile image. The ornamental left band stays
# visible because the chat content begins after the left 24% of the main pane.
# CSS background is intentionally NOT applied to the welcome splash.
CHAT_BG = image_uri('chatbot_bg.png')
WELCOME_BG = image_uri('welcome_bg.jpg')
if not CHAT_BG:
    st.warning('Missing chatbot_bg.png. Put it beside this Python file or inside assets/.')

CSS = r'''
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Sora:wght@500;600;700&family=Noto+Serif+Tibetan:wght@400;600&display=swap');
:root {--cream:#fff5dc;--ink:#3a1d10;--muted:#86624a;--red:#c43c29;--line:rgba(196,60,41,.23);}
html,body,.stApp {font-family:Manrope,system-ui,sans-serif;color:var(--ink);}
.stApp {background-color:var(--cream);background-image:url("__BG__");
 background-position:center center;background-size:100% 100%;background-repeat:no-repeat;background-attachment:fixed;}
header[data-testid="stHeader"] {background:transparent;}
#MainMenu,footer,[data-testid="stDeployButton"] {display:none!important;}
section[data-testid="stSidebar"] {background:rgba(255,246,224,.97);border-right:1px solid var(--line);}
/* Content starts to the RIGHT of the left ornament on desktop. */
[data-testid="stMainBlockContainer"] .block-container {max-width:820px!important;
 width:calc(100% - 24vw)!important;margin-left:clamp(120px,24vw,440px)!important;
 margin-right:auto!important;padding-top:2rem!important;padding-bottom:8rem!important;}
[data-testid="stBottomBlockContainer"] {max-width:820px!important;
 width:calc(100% - 24vw)!important;margin-left:clamp(120px,24vw,440px)!important;margin-right:auto!important;}
[data-testid="stChatMessage"] {background:rgba(255,250,238,.94);border:1px solid var(--line);
 border-radius:17px;padding:14px 18px;margin-bottom:12px;}
[data-testid="stChatMessage"] img {border-radius:50%;object-fit:cover;}
[data-testid="stChatInput"] {background:rgba(255,251,241,.97);border:1px solid var(--line);border-radius:15px;}
.hero h1,.tool-head h1 {font-family:Sora,sans-serif;color:#a92f22;font-size:2rem;margin-bottom:.3rem;}
.hero p,.tool-head p {color:var(--muted);}
.hero .tib,.dz {font-family:'Noto Serif Tibetan','Kailasa',serif;line-height:2;color:var(--ink);}
.hero .tib {color:var(--red);margin-top:10px;}
.dz {font-size:1.22rem;}
.brand {padding:22px 12px;text-align:center;border:1px solid var(--line);border-radius:14px;
 background:rgba(255,250,237,.95);margin-bottom:16px;}
.brand strong {font:700 1.3rem Sora,sans-serif;color:#a92f22;}
.brand small {display:block;color:var(--red);margin-top:4px;}
section[data-testid="stSidebar"] div[data-testid="stButton"] button {border-radius:12px;min-height:45px;}
div[data-testid="stButton"] button[kind="primary"] {background:#b73527;color:white;border-color:#b73527;}
div[data-testid="stButton"] button[kind="primary"]:hover {background:#942d21;color:white;}
[data-testid="stTextArea"] textarea,[data-testid="stTextInput"] input {background:rgba(255,251,241,.97);}
.badge {font-size:.8rem;border-radius:999px;padding:5px 10px;display:inline-block;margin-top:7px;}
.ok {color:#12633e;background:#e3f4e8;}.warn {color:#89530a;background:#fff0c8;}
.card {background:rgba(255,250,238,.96);border:1px solid var(--line);border-radius:15px;padding:20px;}
@media(max-width:1000px) {
 .stApp {background-size:cover;background-position:left center;}
 [data-testid="stMainBlockContainer"] .block-container,
 [data-testid="stBottomBlockContainer"] {width:100%!important;max-width:820px!important;
 margin:0 auto!important;padding-left:1.2rem!important;padding-right:1.2rem!important;}
}
</style>
'''.replace('__BG__', CHAT_BG)


def splash():
    uri = WELCOME_BG or CHAT_BG
    st.markdown(r'''<style>
.stApp {background-image:linear-gradient(to top,rgba(255,245,220,.96),rgba(255,245,220,.55) 35%,transparent 75%),url("__WELCOME__")!important;
 background-size:cover!important;background-position:center!important;}
section[data-testid="stSidebar"] {display:none!important;}
[data-testid="stMainBlockContainer"] .block-container {max-width:760px!important;width:100%!important;
 margin:0 auto!important;min-height:90vh;display:flex;flex-direction:column;
 justify-content:flex-end;text-align:center;padding-bottom:8vh!important;}
</style>'''.replace('__WELCOME__', uri), unsafe_allow_html=True)
    st.markdown('<div style="text-align:center"><div style="font:1.3rem serif;color:#b73527">འབྲུག་རྒྱལ་ཁབ་ཀྱི་རྒྱལ་ཡོངས་སྐད་ཡིག</div>'
                '<h1 style="color:#a92f22">Welcome to Dzongkha AI Assistant</h1>'
                '<p>Ask in English or Dzongkha. Answers come back in Dzongkha.</p></div>', unsafe_allow_html=True)
    if st.button('Get started →', type='primary', use_container_width=True):
        st.session_state.entered = True
        st.rerun()


HISTORY_FILE = ROOT / 'chat_sessions.json'


def load_sessions():
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding='utf-8'))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def write_sessions(sessions):
    try:
        HISTORY_FILE.write_text(json.dumps(sessions, ensure_ascii=False, indent=2), encoding='utf-8')
    except OSError as exc:
        st.warning(f'Could not save chat history: {exc}')


def save_current(kind):
    msgs = st.session_state['hist_pipe' if kind == 'pipe' else 'hist_ft']
    if not msgs:
        return
    key = 'chat_id_' + kind
    sid = st.session_state.get(key) or uuid.uuid4().hex[:10]
    st.session_state[key] = sid
    title = next((m['content'] for m in msgs if m['role'] == 'user'), 'Chat')[:40]
    record = {'id':sid,'kind':kind,'title':title,'ts':time.time(),'messages':msgs,
              'en_hist':st.session_state['en_hist' if kind == 'pipe' else 'en_hist_ft']}
    records = [x for x in load_sessions() if x.get('id') != sid]
    write_sessions([record] + records[:49])


def open_session(sid):
    for record in load_sessions():
        if record.get('id') == sid:
            kind = record.get('kind', 'pipe')
            st.session_state['hist_' + kind] = record.get('messages', [])
            st.session_state['en_hist' if kind == 'pipe' else 'en_hist_ft'] = record.get('en_hist', [])
            st.session_state['chat_id_' + kind] = sid
            st.session_state.active_nav = 'Chat' if kind == 'pipe' else 'Fine-tuned Chat'
            return


def delete_session(sid):
    write_sessions([s for s in load_sessions() if s.get('id') != sid])
    for kind in ('pipe', 'ft'):
        if st.session_state.get('chat_id_' + kind) == sid:
            st.session_state.pop('chat_id_' + kind, None)


def clear_summary():
    st.session_state['sum_text'] = ''
    st.session_state.pop('sum_out', None)


def new_chat():
    for key in ('hist_pipe', 'hist_ft', 'en_hist', 'en_hist_ft'):
        st.session_state[key] = []
    for key in ('chat_id_pipe', 'chat_id_ft', 'pending'):
        st.session_state.pop(key, None)


def render_assistant(message):
    content = message['content']
    if TIBETAN.search(content):
        st.markdown('<div class="dz">' + html.escape(content) + '</div>', unsafe_allow_html=True)
    else:
        st.write(content)
    source = message.get('source')
    badges = {
        'verified':('✔ Verified answer','ok'),
        'machine':('⚠ Machine translation — verify accuracy','warn'),
        'general':('ℹ AI-generated and translated — verify facts','warn'),
    }
    if source in badges:
        title, cls = badges[source]
        st.markdown(f'<span class="badge {cls}">{title}</span>', unsafe_allow_html=True)
    if message.get('notes'):
        with st.expander('Spelling and sentence checks'):
            for note in message['notes']:
                st.write('• ' + note)
    if message.get('debug'):
        with st.expander('Show English steps'):
            st.write('Question:', message['debug'][0])
            st.write('Answer:', message['debug'][1])


def render_history(key):
    for message in st.session_state[key]:
        with st.chat_message(message['role'], avatar=USER_AVATAR if message['role'] == 'user' else BOT_AVATAR):
            if message['role'] == 'user':
                st.write(message['content'])
            else:
                render_assistant(message)


def welcome():
    st.subheader('How can I help you today?')
    st.caption('Ask in English or Dzongkha. Try one of these:')
    prompts = [
        ('👋 Greeting','Hello'),('🙏 Thank you','How do you say thank you in Dzongkha?'),
        ('🏛️ Capital','What is the capital of Bhutan?'),('📅 Tuesday','How do you say Tuesday in Dzongkha?'),
        ('🏔️ Tiger’s Nest','What is Paro Taktsang?'),('🔢 Numbers','How do you say 5 in Dzongkha?'),
    ]
    cols = st.columns(2)
    for i, (title, prompt) in enumerate(prompts):
        if cols[i % 2].button(title, key=f'suggest_{i}', use_container_width=True):
            st.session_state.pending = prompt
            st.rerun()


def chat_header():
    st.markdown('<div class="hero"><h1>Dzongkha AI Assistant</h1>'
                '<p>Ask in English or Dzongkha. Answers come back in Dzongkha.</p>'
                '<div class="tib">འབྲུག་རྒྱལ་ཁབ་ཀྱི་རྒྱལ་ཡོངས་སྐད་ཡིག</div></div>',
                unsafe_allow_html=True)


def tool_header(title, description):
    st.markdown(f'<div class="tool-head"><h1>{html.escape(title)}</h1>'
                f'<p>{html.escape(description)}</p></div>', unsafe_allow_html=True)


for key in ('hist_pipe', 'hist_ft', 'en_hist', 'en_hist_ft'):
    st.session_state.setdefault(key, [])
st.session_state.setdefault('active_nav', 'Chat')
st.session_state.setdefault('sum_text', '')

if not st.session_state.get('entered'):
    splash()
    st.stop()

st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="brand"><strong>Dzongkha AI</strong><small>English ⇄ རྫོང་ཁ</small></div>',
                unsafe_allow_html=True)
    st.caption('WORKSPACE · Choose a tool')
    nav_items = [
        ('Chat', ':material/chat_bubble_outline:'),
        ('Fine-tuned Chat', ':material/school:'),
        ('Summarize', ':material/article:'),
        ('Next Word', ':material/auto_awesome:'),
        ('Translate', ':material/translate:'),
    ]
    for title, icon in nav_items:
        if st.button(title, icon=icon, key='nav_' + title.lower().replace(' ', '_'),
                     type='primary' if st.session_state.active_nav == title else 'secondary',
                     use_container_width=True):
            st.session_state.active_nav = title
            st.rerun()
    if st.button('🗑️ New chat', use_container_width=True):
        new_chat()
        st.rerun()
    st.divider()
    st.caption('HISTORY')
    for rec in load_sessions()[:12]:
        a, b = st.columns([5, 1])
        a.button(rec.get('title', 'Chat'), key='open_' + rec['id'],
                 on_click=open_session, args=(rec['id'],), use_container_width=True)
        b.button('✕', key='del_' + rec['id'], on_click=delete_session, args=(rec['id'],))
    st.divider()
    st.caption('Verified answers come from your answer bank. AI translations should be checked.')

mode = st.session_state.active_nav

if mode in ('Chat', 'Fine-tuned Chat'):
    chat_header()
    kind = 'pipe' if mode == 'Chat' else 'ft'
    key = 'hist_' + kind
    if not st.session_state[key]:
        welcome()
    render_history(key)
    typed = st.chat_input('Message the assistant in English or Dzongkha…')
    question = typed or st.session_state.pop('pending', None)
    if question:
        st.session_state[key].append({'role':'user','content':question})
        with st.chat_message('user', avatar=USER_AVATAR):
            st.write(question)
        with st.chat_message('assistant', avatar=BOT_AVATAR):
            with st.spinner('Thinking…'):
                try:
                    if kind == 'ft' and ADAPTER_PATH.is_dir() and in_training_data(question):
                        TIMES.clear()
                        reply = run_llm('Answer ONLY in Dzongkha, in Tibetan script.',
                                        [{'role':'user','content':question}],
                                        MAX_TOKENS, TEMPERATURE, finetuned=True)
                        if dz_ratio(reply) < 0.6:
                            raise ValueError('Fine-tuned model did not return Dzongkha')
                        source, debug = 'machine', None
                    else:
                        history = st.session_state['en_hist' if kind == 'pipe' else 'en_hist_ft']
                        reply, en_q, en_a, source = pipeline_chat(history, question)
                        debug = (en_q, en_a)
                    notes = []
                    if source != 'verified':
                        reply, notes = check_notes(reply)
                    msg = {'role':'assistant','content':reply,'source':source,
                           'notes':notes,'debug':debug}
                    render_assistant(msg)
                    if TIMES:
                        st.caption('⏱ ' + ' · '.join(f'{k}: {v:.1f}s' for k,v in TIMES.items()))
                    st.session_state[key].append(msg)
                    save_current(kind)
                except Exception as exc:
                    st.error(f'Could not generate a reply: {exc}')
                    st.info('Check Ollama or mlx-lm, your translation model downloads, and installed packages.')

elif mode == 'Translate':
    tool_header('Translate', 'Translate English and Dzongkha directly.')
    direction = st.radio('Translation direction', ['English → Dzongkha', 'Dzongkha → English'],
                         horizontal=True)
    source = st.text_area('Text to translate', height=180,
                          placeholder='Type or paste text…')
    if st.button('Translate now', type='primary'):
        if not source.strip():
            st.warning('Enter some text first.')
        else:
            with st.spinner('Translating…'):
                try:
                    result = (normalize(translate(source.strip(), EN_TO_DZ, EN, DZ, 128))
                              if direction.startswith('English') else
                              translate(source.strip(), DZ_TO_EN, DZ, EN, 128))
                    st.session_state.direct_translation = result
                except Exception as exc:
                    st.error(f'Translation failed: {exc}')
    if st.session_state.get('direct_translation'):
        st.text_area('Translation result', value=st.session_state.direct_translation,
                     height=170)
        st.caption('Machine translation is not guaranteed accurate. Verify important wording.')

elif mode == 'Summarize':
    tool_header('Summarize text', 'Create a short Dzongkha summary of English or Dzongkha text.')
    left, right = st.columns(2, gap='large')
    with left:
        text = st.text_area('Your text', key='sum_text', height=320)
        run = st.button('Summarize', type='primary')
        st.button('Clear summary', on_click=clear_summary)
    if run:
        if not text.strip():
            st.warning('Paste text to summarize.')
        else:
            with st.spinner('Summarizing…'):
                try:
                    source = text.strip()[:4000]
                    en_text = translate(source, DZ_TO_EN, DZ, EN, 160) if dz_ratio(source) >= .5 else source
                    en_summary = clean_en(run_llm('Summarize in one short English sentence (max 25 words).',
                        [{'role':'user','content':en_text}], 64, TEMPERATURE))
                    dz_summary = normalize(translate(en_summary, EN_TO_DZ, EN, DZ, 96))
                    dz_summary, notes = check_notes(dz_summary)
                    st.session_state.sum_out = {'dz':dz_summary,'en':en_summary,'notes':notes}
                except Exception as exc:
                    st.error(f'Summarization failed: {exc}')
    with right:
        st.subheader('Summary in Dzongkha')
        out = st.session_state.get('sum_out')
        if out:
            st.markdown('<div class="card dz">' + html.escape(out['dz']) + '</div>',
                        unsafe_allow_html=True)
            st.code(out['dz'], language=None)
            with st.expander('English summary'):
                st.write(out['en'])
            if out['notes']:
                with st.expander('Spelling checks'):
                    for note in out['notes']:
                        st.write(note)
        else:
            st.info('Your summary will appear here.')

elif mode == 'Next Word':
    tool_header('Next-word prediction', 'Predict likely next words for Dzongkha text.')
    prefix = st.text_input('Start of a Dzongkha sentence', value='འབྲུག་རྒྱལ་ཁབ་')
    if st.button('Predict', type='primary'):
        if not prefix.strip():
            st.warning('Enter Dzongkha text first.')
        else:
            with st.spinner('Predicting…'):
                try:
                    preds = predict_next_words(prefix)
                    for word, probability in preds:
                        st.write(f'{word} — {probability:.1%}')
                except Exception as exc:
                    st.error(f'Next-word prediction failed: {exc}')

st.markdown('<p style="text-align:center;color:#86624a;font-size:.8rem;margin-top:32px">'
            'Dzongkha AI Assistant · NLP project</p>', unsafe_allow_html=True)
