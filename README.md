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
   spellings fixed (ཡིན→ཨིན, གཉིས→གཉིས ...), locative ལུ rule, digits to ༠-༩, foreign characters removed.
3. **Checks shown in the app** - malformed syllables, syllables missing from the word list (with suggestions),
   and verb-final (SOV) sentence-structure warnings.
4. Add real Dzongkha text (one sentence per line) to `data/extra_corpus.txt` to grow the spell-check word list.

## Data note
Zenodo 6271560 is the *Dzongkha Handwritten Digit Dataset* (images of ༠-༩). It is not a text corpus, so it is
used here only for the digit mapping. Cite it (CC-BY 4.0) if you use it.


## GovTech Dzongkha NLP resources

The official GovTech site describes a hosted Dzongkha-English NLLB translation
model fine-tuned on approximately 700,000 DCDD-provided parallel sentences:
https://dnlp.tech.gov.bt/about

**Important:** the public About page documents the model and corpus size, but it
does not provide a public corpus download or a documented API endpoint. This
repository does not scrape the service or include the DCDD corpus. Request the
dataset/API access and usage permission from GovTech/DCDD before using it.

This project adds the public `dzoseg` tokenizer (trained on a separate Dzongkha
web/literary corpus) for optional segmentation and text diagnostics. It is not
the DCDD 700K dataset and is not a replacement for the GovTech translation model.

### Prepare an authorised parallel corpus

If GovTech/DCDD provides you a corpus file, place it in `data/` and run:

```bash
python prepare_govtech_corpus.py --input data/your_authorised_corpus.csv \
  --english-column english --dzongkha-column dzongkha
```

Supported inputs: CSV, TSV, JSON, JSONL and XLSX. The converter validates that
the target side contains Tibetan-script Dzongkha and removes exact duplicates.
For model fine-tuning, the generated JSONL can be passed to the existing trainer for a data check:

```bash
python train_en2dz.py --jsonl data/govtech_parallel.jsonl --dry-run
```

Training the 600M NLLB model is resource-intensive; use a suitable GPU runtime
and review licensing/permission before training or redistributing any corpus.
\n\nSee [CST_RESOURCE_NOTES.md](CST_RESOURCE_NOTES.md) for what is publicly available from CST and how to import an authorised corpus. The verified phrase bank now returns `ཀུ་ཟུ་ཟང་པོ་ལ།` for Hello and `བཀའ་དྲིན་ཆེ།` for Thank you.\n