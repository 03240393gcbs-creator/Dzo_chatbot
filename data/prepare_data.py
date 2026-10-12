import json, random, os

random.seed(42)
data = json.load(open("dataset.json", encoding="utf-8"))
random.shuffle(data)

rows = []
for d in data:
    user_text = d["instruction"]
    if d.get("input"):
        user_text += "\n" + d["input"]
    rows.append({"messages": [
        {"role": "system", "content": d.get("system", "Always reply in Dzongkha.")},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": d["output"]},
    ]})

split = max(1, int(len(rows) * 0.1))   # 10% for checking
valid, train = rows[:split], rows[split:]

os.makedirs("data", exist_ok=True)
for name, part in (("train", train), ("valid", valid)):
    with open(f"data/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"train: {len(train)}, valid: {len(valid)}")