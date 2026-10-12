"""Fine-tune NLLB for English -> Dzongkha on a parallel corpus.

Your data can be either:
  A) a CSV/TSV with an English column and a Dzongkha column
       python train_en2dz.py --table pairs.csv --en-col english --dz-col dzongkha
  B) two plain-text files, one sentence per line, line N matching line N
       python train_en2dz.py --en-file train.en --dz-file train.dz

Check your data first (no training, no big downloads):
       python train_en2dz.py --table pairs.csv --en-col english --dz-col dzongkha --dry-run

Training a 600M model is heavy. Use a free Google Colab GPU, or on a MacBook Air
use a small --max-pairs (e.g. 5000) and --batch-size 4 as a test run.
"""
import argparse, csv, random, re, sys

TIB = re.compile(r"[\u0F00-\u0FFF]")


def read_pairs(a):
    pairs = []
    if a.table:
        delim = "\t" if a.table.endswith((".tsv", ".txt")) else ","
        with open(a.table, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter=delim):
                pairs.append((row[a.en_col], row[a.dz_col]))
    elif a.en_file and a.dz_file:
        with open(a.en_file, encoding="utf-8") as fe, open(a.dz_file, encoding="utf-8") as fd:
            en, dz = fe.read().splitlines(), fd.read().splitlines()
        if len(en) != len(dz):
            sys.exit(f"Line counts differ: {len(en)} English vs {len(dz)} Dzongkha. Files are not aligned.")
        pairs = list(zip(en, dz))
    else:
        sys.exit("Give --table, or both --en-file and --dz-file.")
    return pairs


def clean(pairs, max_chars=300):
    seen, out = set(), []
    for en, dz in pairs:
        en, dz = " ".join(str(en).split()), " ".join(str(dz).split())
        if len(en) < 3 or len(dz) < 3 or len(en) > max_chars or len(dz) > max_chars * 3:
            continue
        letters = [c for c in dz if c.isalpha() or TIB.match(c)]
        if not letters or len(TIB.findall(dz)) / len(letters) < 0.5:   # Dzongkha side must be Tibetan script
            continue
        if re.fullmatch(r"[A-Za-z0-9\s\W]*", en) is None and len(TIB.findall(en)) > 0:  # English side contains Tibetan: swapped columns
            continue
        if (en, dz) in seen:
            continue
        seen.add((en, dz))
        out.append((en, dz))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table"); p.add_argument("--en-col", default="english"); p.add_argument("--dz-col", default="dzongkha")
    p.add_argument("--en-file"); p.add_argument("--dz-file")
    p.add_argument("--base", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--out", default="./models/nllb-en2dz")
    p.add_argument("--max-pairs", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    raw = read_pairs(a)
    pairs = clean(raw)
    print(f"read {len(raw)} pairs, kept {len(pairs)} after cleaning")
    random.seed(0); random.shuffle(pairs)
    pairs = pairs[: a.max_pairs + 300]
    test, train = pairs[:300], pairs[300:]
    print(f"train {len(train)}  test {len(test)}")
    for en, dz in train[:3]:
        print("  EN:", en, "\n  DZ:", dz)
    if a.dry_run:
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from transformers.optimization import Adafactor

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)
    tok = AutoTokenizer.from_pretrained(a.base, src_lang="eng_Latn", tgt_lang="dzo_Tibt")
    model = AutoModelForSeq2SeqLM.from_pretrained(a.base).to(device)
    opt = Adafactor(model.parameters(), lr=a.lr, scale_parameter=False, relative_step=False, warmup_init=False)

    model.train()
    step = 0
    for ep in range(a.epochs):
        random.shuffle(train)
        for i in range(0, len(train), a.batch_size):
            batch = train[i:i + a.batch_size]
            enc = tok([b[0] for b in batch], text_target=[b[1] for b in batch],
                      return_tensors="pt", padding=True, truncation=True, max_length=128)
            enc["labels"][enc["labels"] == tok.pad_token_id] = -100
            enc = {k: v.to(device) for k, v in enc.items()}
            loss = model(**enc).loss
            loss.backward()
            opt.step(); opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"epoch {ep+1} step {step} loss {loss.item():.3f}", flush=True)

    model.eval()
    tok.save_pretrained(a.out); model.save_pretrained(a.out)
    print("saved to", a.out)

    # quick check on held-out pairs
    hyps = []
    with torch.no_grad():
        for en, _ in test[:100]:
            ids = tok(en, return_tensors="pt", truncation=True, max_length=128).to(device)
            out = model.generate(**ids, forced_bos_token_id=tok.convert_tokens_to_ids("dzo_Tibt"), max_new_tokens=96, num_beams=4)
            hyps.append(tok.batch_decode(out, skip_special_tokens=True)[0])
    try:
        import sacrebleu
        print("BLEU on 100 held-out pairs:", sacrebleu.corpus_bleu(hyps, [[d for _, d in test[:100]]]).score)
    except ImportError:
        print("pip install sacrebleu to get a BLEU score. Samples:")
    for (en, ref), hyp in list(zip(test, hyps))[:5]:
        print("EN :", en, "\nREF:", ref, "\nHYP:", hyp, "\n")


if __name__ == "__main__":
    main()