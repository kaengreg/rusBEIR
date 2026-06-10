# %%
import json
from pathlib import Path

# %%
corpus_orig = 'corpus.jsonl'
corpus_part_trans = 'corpus-v2.jsonl'
output_file = 'corpus-rest.jsonl'

ids_file2 = set()
with open(corpus_part_trans, 'r', encoding='utf-8') as f2:
    for line in f2:
        line = line.strip()
        if line:    
            record = json.loads(line)
            ids_file2.add(record['_id'])

Path(output_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
with open(corpus_orig, 'r', encoding='utf-8') as f1, \
    open(output_file, 'w', encoding='utf-8') as out:
    for line in f1:
        line = line.strip()
        if line: 
            record = json.loads(line)
            if record['_id'] not in ids_file2:
                out.write(json.dumps(record, ensure_ascii=False) + '\n')    
    
print('Done!')        

# %%


