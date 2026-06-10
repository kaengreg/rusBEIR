# %%
import json
from pathlib import Path

# %%
def get_empty_text_ids(input_file):
    empty_ids = set()

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            if doc.get('text') == "" or doc.get("title") == "":
                empty_ids.add(doc['_id'])
    return empty_ids

def filter_docs(input_file, output_file, empty_ids):
    Path(output_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            doc = json.loads(line)
            if doc['_id'] in empty_ids:
                fout.write(json.dumps(doc, ensure_ascii=False) + '\n')
        

# %%
file1 = 'trans_part.jsonl'
file2 = 'orig_corpus.jsonl'
output_file = 'corpus-rest.jsonl'
empty_ids = get_empty_text_ids(file1)
filter_docs(file2, output_file, empty_ids)

# %%


