import json, random, pathlib

SRC = "dzo_dataset_train.json"
OUT = pathlib.Path("data/mlx")
SYSTEM = "Always reply in Dzongkha."

rows = json.load(open(SRC, encoding="utf-8"))

HELLO = "ཀུ་ཟུ་ཟང་ཕོ་ལ།"
for q in ["hello", "Hello", "hello!", "hi", "Hi", "hi there", "hey", "Hello, how are you?"]:
    rows.append({"instruction": q, "input": "", "output": HELLO, "system": SYSTEM})

seen, uniq = set(), []
for r in rows:
    key = (r["instruction"].strip().lower(), r["input"])
    if key not in seen:
        seen.add(key)
        uniq.append(r)

def to_chat(r):
    user = r["instruction"] + (("\n" + r["input"]) if r["input"] else "")
    return {"messages": [
        {"role": "system", "content": r.get("system") or SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": r["output"]},
    ]}

random.seed(0)
random.shuffle(uniq)
n_valid = max(4, len(uniq) // 10)
valid, train = uniq[:n_valid], uniq[n_valid:]

OUT.mkdir(parents=True, exist_ok=True)
for name, part in [("train", train), ("valid", valid)]:
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(to_chat(r), ensure_ascii=False) + "\n")

print(f"train: {len(train)}  valid: {len(valid)}  -> {OUT}/")
