# Dzo_chatbot

English/Dzongkha chatbot (NLP module assignment).

## Run
    pip install -r requirements.txt
    python fix_dataset.py          # clean data, rebuild data/mlx and data/lexicon.txt
    python -m pytest -q            # tests for the Dzongkha tools
    streamlit run app.py

## How Dzongkha quality is controlled
1. **verified_kb.json** - hand-checked question -> Dzongkha answers. Matches are returned word for word.
2. **dzongkha_nlp.py** - every other reply is cleaned: no spaces between syllables, one shad, wrong
   spellings fixed (ཡིན→ཨིན, ཉིས→གཉིས ...), locative ལུ rule, digits to ༠-༩, foreign characters removed.
3. **Checks shown in the app** - malformed syllables, syllables missing from the word list (with suggestions),
   and verb-final (SOV) sentence-structure warnings.
4. Add real Dzongkha text (one sentence per line) to `data/extra_corpus.txt` to grow the spell-check word list.

## Data note
Zenodo 6271560 is the *Dzongkha Handwritten Digit Dataset* (images of ༠-༩). It is not a text corpus, so it is
used here only for the digit mapping. Cite it (CC-BY 4.0) if you use it.
