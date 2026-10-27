"""Clean every Dzongkha answer in your dataset files and rebuild the MLX training files.

    python fix_dataset.py

Reads : dzo_dataset_full.json, dzo_dataset_train.json, dzo_dataset.csv, verified_kb.json
Writes: the same files (cleaned, backups saved as *.bak), data/mlx/train.jsonl, valid.jsonl,
        data/lexicon.txt (syllable list used by the spell checker),
        and a printed change report you can paste into your assignment.
"""
import csv, json, pathlib, random, shutil
from dzongkha_nlp import analyse, normalize

SYSTEM = "Always reply in Dzongkha."
changes_log = []
_kb = json.loads(pathlib.Path("verified_kb.json").read_text(encoding="utf-8"))
def _k(q): return q.strip().lower().rstrip("?!. ")
KB_ANS = {_k(q): it["dz_answer"] for it in _kb for q in it["questions"]}


def clean_answer(text, where, question=None):
    orig = text
    if question and _k(question) in KB_ANS:      # the verified answer bank wins
        text = KB_ANS[_k(question)]
    r = analyse(text)
    if r.fixed != orig or r.changes:
        changes_log.append((where, orig, r.fixed, r.changes))
    return r.fixed


def backup(p):
    p = pathlib.Path(p)
    if p.exists() and not pathlib.Path(str(p) + ".bak").exists():
        shutil.copy(p, str(p) + ".bak")


# 1. JSON datasets
for name in ["dzo_dataset_full.json", "dzo_dataset_train.json"]:
    p = pathlib.Path(name)
    if not p.exists():
        continue
    backup(p)
    rows = json.loads(p.read_text(encoding="utf-8"))
    for r in rows:
        r["output"] = clean_answer(r["output"], f"{name}: {r['instruction']}", r["instruction"])
    p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

# 2. CSV
p = pathlib.Path("dzo_dataset.csv")
if p.exists():
    backup(p)
    with open(p, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f)); fields = rows[0].keys()
    for r in rows:
        r["output"] = clean_answer(r["output"], f"csv: {r['instruction']}", r["instruction"])
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields)); w.writeheader(); w.writerows(rows)

# 3. training pairs = verified answer bank (every question phrasing -> same checked answer)
kb = json.loads(pathlib.Path("verified_kb.json").read_text(encoding="utf-8"))
pairs = []
for it in kb:
    for q in it["questions"]:
        pairs.append((q, it["dz_answer"]))
seen, uniq = set(), []
for q, a in pairs:
    if q.lower() not in seen:
        seen.add(q.lower()); uniq.append((q, a))
random.seed(0); random.shuffle(uniq)
n_valid = max(4, len(uniq) // 10)
valid, train = uniq[:n_valid], uniq[n_valid:]
out = pathlib.Path("data/mlx"); out.mkdir(parents=True, exist_ok=True)
for nm, part in [("train", train), ("valid", valid)]:
    with open(out / f"{nm}.jsonl", "w", encoding="utf-8") as f:
        for q, a in part:
            f.write(json.dumps({"messages": [{"role": "system", "content": SYSTEM},
                                             {"role": "user", "content": q},
                                             {"role": "assistant", "content": a}]}, ensure_ascii=False) + "\n")

# 4. lexicon for the spell checker
from dzongkha_nlp import Lexicon, AnswerBank
lex = Lexicon()
AnswerBank("verified_kb.json").build_lexicon(lex)
pathlib.Path("data/lexicon.txt").write_text("\n".join(f"{s}" for s in sorted(lex.counts)), encoding="utf-8")

print(f"train {len(train)}  valid {len(valid)}  lexicon syllables {len(lex)}")
print(f"\n{len(changes_log)} answers changed:")
for where, old, new, ch in changes_log:
    print(f"- {where}\n    before: {old}\n    after : {new}\n    fixes : {ch}")
