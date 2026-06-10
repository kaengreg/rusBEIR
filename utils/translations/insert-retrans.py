# %%
import json
from pathlib import Path

# %%
def read_jsonl(file_path, key_field = '_id'):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            body = json.loads(line)
            key = body.get(key_field) 
            data[key] = body

    return data

def save_dicts_to_jsonl(data, filename):
    Path(filename).expanduser().parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# %%
def insert_retrans_in_corpus(corpus_file, retrans_file):
    retrans = read_jsonl(retrans_file)
    corpus = read_jsonl(corpus_file)

    for el in corpus:
        if corpus[str(el)]['text'] == "":
            corpus[str(el)]['text'] = retrans[el]['text']
        if corpus[str(el)]['title'] == "":
            corpus[str(el)]['title'] = retrans[el]['title']

    
    save_dicts_to_jsonl(corpus.values(), corpus_file)

# %%
insert_retrans_in_corpus('corpus.jsonl', 
                         'corpus-retrans.jsonl')

# %%


