from datasets import load_dataset
import pandas as pd
from rusBeIR.beir.datasets.data_loader_hf import HFDataLoader
from rusBeIR.benchmarking.model_benchmark import DatasetEvaluator
from rusBeIR.retrieval.models.E5Model import E5Model
from rusBeIR.beir.retrieval.evaluation import EvaluateRetrieval
import json
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from typing import List, Dict
import os
from rusBeIR.retrieval.models.HFTransformers import HFTransformers
import csv
from pathlib import Path


class QueryDataset(Dataset):
    def __init__(self, queries, queries_ids):
        self.queries = queries
        self.query_ids = queries_ids

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.query_ids[idx]


class E5ModelTuned(E5Model):
    def __init__(self, model_name: str = 'intfloat/multilingual-e5-large', maxlen: int = 512, batch_size: int = 128,
                 device: str = 'cuda'):
        super().__init__(model_name, maxlen=maxlen, batch_size=batch_size, device=device)

    def retrieve(self, queries: Dict[str, str], corpus_emb: np.ndarray, corpus_ids: List[str],
                 top_n: int = 100) -> Dict[str, Dict[str, float]]:

        batch_size = 16
        num_workers = 4
        top_n = top_n

        query_dataset = QueryDataset(list(queries.values()), list(queries.keys()))
        data_loader = DataLoader(query_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

        results = {}

        for batch_queries, batch_query_ids in tqdm(data_loader, desc="Processing Queries"):
            query_embs = self.encode_passages(batch_queries)
            similarities = cosine_similarity(query_embs, corpus_emb)

            for idx, query_id in enumerate(batch_query_ids):
                query_similarities = similarities[idx].flatten()
                top_n_indices = query_similarities.argsort()[-top_n:][::-1]
                top_n_results = {corpus_ids[j]: float(query_similarities[j]) * 100 for j in top_n_indices}
                results[query_id] = top_n_results
        return results


corpus, queries, qrels = HFDataLoader(hf_repo="BeIR/msmarco", hf_repo_qrels="BeIR/msmarco-qrels", streaming=False,
                                       keep_in_memory=False, text_type='text').load(split='validation')

nested_keys = [key for sublist in qrels.values() for key in sublist.keys()]
rel_docs = {cid: corpus[cid]['text'] for cid in nested_keys}

e5tuned = E5ModelTuned()
corpus_emb = e5tuned.encode_passages([doc['text'] for doc in corpus.values()])
results = e5tuned.retrieve(rel_docs, corpus_emb, list(corpus.keys()), top_n=10)

expanded_jsonl = 'msmarco-qrels-expand.jsonl'
Path(expanded_jsonl).expanduser().parent.mkdir(parents=True, exist_ok=True)
with open(expanded_jsonl, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

new_rel_docs = {cid: {k: 1 for k, v in cres.items() if v > 96} for cid, cres in list(results.items())}

updated_qrels = {}

for query_id, doc_scores in qrels.items():
    updated_docs = {}
    for doc_id, score in doc_scores.items():
        if doc_id in new_rel_docs:
            for new_doc_id, new_score in new_rel_docs[doc_id].items():
                updated_docs[new_doc_id] = score * new_score
        else:
            updated_docs[doc_id] = score
    updated_qrels[query_id] = updated_docs

output_file = "dev-expanded-qp.tsv"

Path(output_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(['query-id', 'corpus-id', 'score'])

    for query_id, doc_scores in updated_qrels.items():
        for doc_id, score in doc_scores.items():
            writer.writerow([query_id, doc_id, score])

print(f"Данные успешно записаны в файл {output_file}.")
