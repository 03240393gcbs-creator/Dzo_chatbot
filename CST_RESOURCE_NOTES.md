# CST Dzongkha NLP resource integration notes

Project website: https://nlp.cst.edu.bt/

## What the public CST website exposes

The site provides three hosted tools: Dzongkha-English neural machine translation,
Dzongkha automatic speech recognition, and Dzongkha text-to-speech. The translation
page says its NLLB-200-distilled-600M model was fine-tuned on a 30,000-sentence
parallel corpus. The public page does not provide a downloadable copy of that
corpus or a documented public inference API.

Therefore this repository does not scrape the website, copy its private corpus,
or pretend the 30K data is included. Ask CST/DCDD for the corpus/model/API and
the licence/permission before training on or redistributing it.

## Current chatbot behaviour

- English and Dzongkha input are accepted by the chat pipeline.
- The hand-checked `verified_kb.json` answer bank is checked before machine
  translation. Common phrases such as Hello and Thank you are returned exactly.
- Other questions use the configured English/Dzongkha translation models and
  are marked as machine translated. Rule-based cleanup is not a substitute for
  review by a fluent Dzongkha speaker.
- `govtech_dnlp.py` can validate a corpus file if CST/DCDD provides one in CSV,
  TSV, JSON, JSONL or XLSX format.

## Import an authorised corpus

Once CST/DCDD gives you a file and permission to use it, place it under `data/`
and run (adjust column names if needed):

```bash
python prepare_govtech_corpus.py --input data/cst_parallel.csv \
  --english-column english --dzongkha-column dzongkha \
  --output data/cst_parallel.jsonl
python train_en2dz.py --jsonl data/cst_parallel.jsonl --dry-run
```

Actual fine-tuning is a separate, resource-intensive step. Do not run it on a
MacBook Air expecting it to train the full 600M model quickly; use a suitable
GPU runtime and confirm the corpus licence first.

## Phrase spellings used in the verified answer bank

- Hello / greeting: `ཀུ་ཟུ་ཟང་པོ་ལ།` (Kuzuzangpo la)
- Thank you: `བཀའ་དྲིན་ཆེ།`

The phrase `སྐུ་གཟུགས་བཟང་པོ` is not the standard standalone greeting
“Hello”; it is not used as the answer-bank output. Please have a fluent
Dzongkha speaker review formal or domain-specific sentences before deployment.
