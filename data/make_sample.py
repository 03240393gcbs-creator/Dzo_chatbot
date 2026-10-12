import json

pairs = [
    ("Where is the Tiger's Nest monastery located?", "སྤ་རོ་སྟག་ཚང་འདི་ སྤ་རོ་རྫོང་ཁག་ནང་ཆགས་ཏེ་ཡོད།"),
    ("ཁྱོད་ཀྱི་མིང་ག་ཅི་སྨོ།", "ང་གློག་རིག་ཐབས་རིག་གི་རོགས་རམ་པ་ཅིག་ཨིན།"),
]

# repeat the 2 pairs so the test has enough rows to split
data = []
for i in range(10):
    for q, a in pairs:
        data.append({"instruction": q, "input": "", "output": a,
                     "system": "Always reply in Dzongkha."})

json.dump(data, open("dataset.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(len(data), "rows written")