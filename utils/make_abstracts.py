from datasets import load_dataset
import pandas as pd
import json
from pathlib import Path

dataset = load_dataset('ai-forever/libra', 'long_context_multiq')
split = 'test'
df = dataset[split].to_pandas()

df.insert(0, "_id", [i for i in range(len(df))])

def split_into_paragraphs(example):
    paragraphs = [p.strip() for p in example['context'].split('\n\n') if len(p.strip()) > 50]
    return [{"_id": f"{example['_id']}/{idx}", "context": paragraph, "input": example['input']}
            for idx, paragraph in enumerate(paragraphs)]


expanded_dataset = [split_into_paragraphs(doc) for doc in df.to_dict(orient='records')]
expanded_ds = pd.DataFrame([item for sublist in expanded_dataset for item in sublist])

corpus_dict, queries_dict, tsv_pairs = {}, {}, ['query_id\tcorpus_id\tscore']
text_id_map, query_id_map = {}, {}
next_corpus_id, next_query_id = 0, 100

for _, item in expanded_ds.iterrows():
    text, question = item["context"], item["input"]

    if text not in text_id_map:
        text_id_map[text] = next_corpus_id
        corpus_dict[next_corpus_id] = {"_id": next_corpus_id, "title": "", "text": text}
        next_corpus_id += 1
    corpus_id = text_id_map[text]

    if question not in query_id_map:
        query_id_map[question] = next_query_id
        queries_dict[next_query_id] = {"_id": next_query_id, "text": question}
        next_query_id += 1
    query_id = query_id_map[question]

    tsv_pairs.append(f"{query_id}\t{corpus_id}\t1")

corpus_output_file = 'corpus.jsonl'
queries_output_file = 'queries.jsonl'
qrels_output_file = 'test.tsv'

Path(corpus_output_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
with open(corpus_output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(json.dumps(entry, ensure_ascii=False) for entry in corpus_dict.values()) + '\n')

Path(queries_output_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
with open(queries_output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(json.dumps(entry, ensure_ascii=False) for entry in queries_dict.values()) + '\n')


"""
In case if qrels could be done this way

Path(qrels_output_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
with open(qrels_output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(tsv_pairs) + '\n')
"""
