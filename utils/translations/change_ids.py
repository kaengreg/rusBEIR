# %%
import json
import pandas as pd
from pathlib import Path

# %%
def add_prefix_to_ids(input_file, output_file, prefix):
    Path(output_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
    with open(input_file, 'r', encoding='utf-8') as fin, \
        open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            record = json.loads(line)
            record['_id'] = prefix + record['_id']
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')

# %%
input_file = 'corpus.jsonl'
output_file = 'corpus-renumbered.jsonl'
prefix = 'unix-cor-'
add_prefix_to_ids(input_file, output_file, prefix)


# %% [markdown]
# ---

# %%
def add_prefix_to_tsv(input_file, output_file, query_prefix, corpus_prefix):
   df = pd.read_csv(input_file, sep='\t', header=0, names=['query-id', 'corpus-id', 'score'])

   df['query-id'] = query_prefix + df['query-id'].astype(str)
   df['corpus-id'] = corpus_prefix + df['corpus-id'].astype(str)

   Path(output_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
   df.to_csv(output_file, sep='\t', header=True, index=False)

# %%
input_file = 'test.tsv'
output_file =  'test_unix_renumbered.tsv'
query_prefix = 'unix-que-'
corpus_prefix = 'unix-cor-'  

add_prefix_to_tsv(input_file, output_file, query_prefix, corpus_prefix)

# %%


