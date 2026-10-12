import json
S = "Always reply in Dzongkha."
pairs = [
 ("Where is the Tiger's Nest monastery located?", "སྤ་རོ་སྟག་ཚང་འདི་ སྤ་རོ་རྫོང་ཁག་ནང་ཆགས་ཏེ་ཡོད།"),
 ("ཁྱོད་ཀྱི་མིང་ག་ཅི་སྨོ།", "ང་གློག་རིག་ཐབས་རིག་གི་རོགས་རམ་པ་ཅིག་ཨིན།"),
 ("How do you say hello in Dzongkha?", "ཀུ་ཟུ་ཟང་ཕོ་ལ།"),
 ("Hello", "ཀུ་ཟུ་ཟང་ཕོ་ལ།"),
 ("How do you say thank you in Dzongkha?", "བཀའ་དྲིན་ཆེ།"),
 ("Thank you", "བཀའ་དྲིན་ཆེ།"),
 ("How do you say yes in Dzongkha?", "ཨིན།"),
 ("How do you say no in Dzongkha?", "མེན།"),
 ("How do you say 1 in Dzongkha?", "གཅིག།"),
 ("How do you say 2 in Dzongkha?", "ཉིས།"),
 ("How do you say 3 in Dzongkha?", "སུམ།"),
 ("How do you say 4 in Dzongkha?", "བཞི།"),
 ("How do you say 5 in Dzongkha?", "ལྔ།"),
 ("How do you say 6 in Dzongkha?", "དྲུག།"),
 ("How do you say 7 in Dzongkha?", "བདུན།"),
 ("How do you say 8 in Dzongkha?", "བརྒྱད།"),
 ("How do you say 9 in Dzongkha?", "དགུ།"),
 ("How do you say 10 in Dzongkha?", "བཅུ།"),
 ("How do you say Monday in Dzongkha?", "གཟའ་ཟླ་བ།"),
 ("How do you say Tuesday in Dzongkha?", "གཟའ་མིག་དམར།"),
 ("How do you say Wednesday in Dzongkha?", "གཟའ་ལྷག་པ།"),
 ("How do you say Thursday in Dzongkha?", "གཟའ་ཕུར་བུ།"),
 ("How do you say Friday in Dzongkha?", "གཟའ་པ་སངས།"),
 ("How do you say Saturday in Dzongkha?", "གཟའ་སྤེན་པ།"),
 ("How do you say Sunday in Dzongkha?", "གཟའ་ཉི་མ།"),
 ("What is the capital of Bhutan?", "འབྲུག་གི་རྒྱལ་ས་འདི་ ཐིམ་ཕུ་ཨིན།"),
 ("What is the national language of Bhutan?", "འབྲུག་གི་རྒྱལ་ཡོངས་སྐད་ཡིག་འདི་ རྫོང་ཁ་ཨིན།"),
 ("What is Paro Taktsang?", "སྤ་རོ་སྟག་ཚང་འདི་ སྤ་རོ་རྫོང་ཁག་ནང་ལུ་ཡོད་པའི་ དགོན་སྡེ་ཅིག་ཨིན།"),
]
rows = [{"instruction": q, "input": "", "output": a, "system": S} for q, a in pairs]
json.dump(rows, open("dzo_dataset_train.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("wrote", len(rows), "rows")
