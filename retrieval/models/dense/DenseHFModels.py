import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from typing import List, Dict
import os
import json
import gc 
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class QueryDataset(Dataset):
    def __init__(self, queries: Dict[str, Dict[str, str]]):
        self.queries = queries

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query_id = list(self.queries.keys())[idx]
        query_text = self.queries[query_id]
        return query_text, query_id


class DenseHFModels:
    def __init__(self, model_name: str, maxlen: int = 512, batch_size: int = 128, device: str = 'cuda',
                 model_sep: str = "\n", padding_side: str | None = None):
        """
        initialize model and tokenizer from hf-transformers
        :param model_name: hf-model repo
        :param device: where to run the model
        """
        self.model, self.tokenizer = self.load_model(model_name, device)
        if padding_side is not None:
            self.tokenizer.padding_side = padding_side
        self.max_len = maxlen
        self.device = device
        self.batch_size = batch_size
        self.model_sep = model_sep

    def load_model(self, model_name: str, device: str = 'cuda'):
        model = AutoModel.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()

        return model, tokenizer

    def encode_queries(self, queries: Dict[str, str], pooling_method: str = "average", prefix: str = ''):
        """
        Encodes queries
        :param queries: list of queries to encode
        :param batch_size: batch_size for encoding
        :return: queries embedding
        """
        queries = [prefix + query for query in queries.values()]
        return self._get_embeddings(queries, pooling_method=pooling_method)
    

    def encode_corpus(self, corpus: Dict[str, Dict[str, str]], pooling_method: str = "average", prefix: str = ''):
        """
        Encodes passages
        :param passages: list of passages to encode
        :param batch_size: batch_size for encoding
        :return:
        """
        corpus = [prefix + doc['title'] + self.model_sep + doc['text'] for doc in corpus.values()]
        return self._get_embeddings(corpus, pooling_method=pooling_method)
    

    def iter_corpus_batches(self, corpus: Dict[str, Dict[str, str]], batch_size: int):
        items = list(corpus.items())
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            ids = [doc_id for doc_id, _ in batch]
            docs = {doc_id: doc for doc_id, doc in batch}
            yield ids, docs

    def resolve_faiss_device(self, faiss_module, faiss_device: str, faiss_gpu_id: int) -> str:
        if faiss_device == "cpu":
            return "cpu"
        if faiss_device == "cuda":
            if not hasattr(faiss_module, "StandardGpuResources"):
                raise RuntimeError("FAISS GPU support is not available. Install a GPU-enabled FAISS build or use --faiss-device cpu.")
            return "cuda"
        if hasattr(faiss_module, "StandardGpuResources"):
            return "cuda"
        return "cpu"

    def faiss_to_gpu(self, faiss_module, index, faiss_device: str, faiss_gpu_id: int):
        resolved_device = self.resolve_faiss_device(faiss_module, faiss_device, faiss_gpu_id)
        if resolved_device == "cpu":
            return index
        resources = faiss_module.StandardGpuResources()
        return faiss_module.index_cpu_to_gpu(resources, faiss_gpu_id, index)

    def faiss_to_cpu(self, faiss_module, index):
        if hasattr(faiss_module, "index_gpu_to_cpu"):
            try:
                return faiss_module.index_gpu_to_cpu(index)
            except RuntimeError:
                return index
        return index

    def build_load_faiss_index(self, corpus: Dict[str, Dict[str, str]], index_dir: str | Path, index_name: str,
                               corpus_batch_size: int = 50000, rebuild_index: bool = False,
                               faiss_device: str = "auto", faiss_gpu_id: int = 0):
        import faiss 
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        index_path = index_dir/f"{index_name}.faiss"
        doc_ids_path = index_dir/f"{index_name}.doc_ids.json"

        if index_path.exists() and doc_ids_path.exists() and not rebuild_index:
            index = faiss.read_index(str(index_path))
            doc_ids = json.loads(doc_ids_path.read_text(encoding="utf-8"))
            index = self.faiss_to_gpu(faiss, index, faiss_device, faiss_gpu_id)
            return index, doc_ids
        
        index = None 
        doc_ids = []

        for batch_ids, batch_docs in self.iter_corpus_batches(corpus, corpus_batch_size):
            embeddings = self.encode_corpus(batch_docs).astype("float32")

            if index is None:
                dim = embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                index = self.faiss_to_gpu(faiss, index, faiss_device, faiss_gpu_id)

            index.add(embeddings)
            doc_ids.extend(batch_ids)

            del embeddings
            gc.collect()

        if index is None:
            raise ValueError("Cannot build FAISS index from an empty corpus.")

        index_to_save = self.faiss_to_cpu(faiss, index)
        faiss.write_index(index_to_save, str(index_path))
        doc_ids_path.write_text(json.dumps(doc_ids, ensure_ascii=False), encoding="utf-8")

        return index, doc_ids
    
    def retrieve_faiss(self, queries: Dict[str, str], corpus: Dict[str, Dict[str, str]], top_n: int = 100, query_batch_size: int = 16, 
                       corpus_batch_size: int = 50000, index_dir: str | Path = "leaderboard/faiss_indexes", index_name: str = "index",
                       rebuild_index: bool = False, faiss_device: str = "auto", faiss_gpu_id: int = 0):
        index, doc_ids = self.build_load_faiss_index(
            corpus=corpus,
            index_dir=index_dir,
            index_name=index_name,
            corpus_batch_size=corpus_batch_size,
            rebuild_index=rebuild_index,
            faiss_device=faiss_device,
            faiss_gpu_id=faiss_gpu_id,
        )

        query_dataset = QueryDataset(queries)
        data_loader = DataLoader(query_dataset, batch_size=query_batch_size, num_workers=0, pin_memory=False)

        results = {}

        for batch_queries, batch_query_ids in tqdm(data_loader, desc="FAISS Search Queries"):
            query_dict = {query_id: query for query_id, query in zip(batch_query_ids, batch_queries)}  
            query_embs = self.encode_queries(query_dict).astype("float32")
            scores, indices = index.search(query_embs, top_n)
            
            for row_idx, query_id in enumerate(batch_query_ids):
                query_results = {}
                for score, doc_idx in zip(scores[row_idx], indices[row_idx]):
                    if doc_idx < 0:
                        continue
                    query_results[doc_ids[doc_idx]] = float(score) * 100
                results[str(query_id)] = query_results
        return results

    def retrieve(self, queries: Dict[str, str], 
               corpus: Dict[str, Dict[str, str]],
               top_n: int = 100, 
               data_batch_size: int = 16, 
               num_workers = 4
               ) -> Dict[str, Dict[str, float]]:

        corpus_emb = self.encode_corpus(corpus)
        corpus_ids = list(corpus.keys())

        data_batch_size = data_batch_size
        num_workers = num_workers
        top_n = top_n

        query_dataset = QueryDataset(queries)
        data_loader = DataLoader(query_dataset, batch_size=data_batch_size, num_workers=num_workers, pin_memory=True)

        results = {}

        for batch_queries, batch_query_ids in tqdm(data_loader, desc="Processing Queries"):
            query_dict = {query_id: query for query_id, query in zip(batch_query_ids, batch_queries)}

            query_embs = self.encode_queries(query_dict)

            similarities = cosine_similarity(query_embs, corpus_emb)

            for idx, query_id in enumerate(batch_query_ids):
                query_similarities = similarities[idx].flatten()
                top_n_indices = query_similarities.argsort()[-top_n:][::-1]
                top_n_results = {corpus_ids[j]: float(query_similarities[j]) * 100 for j in top_n_indices}
                results[query_id] = top_n_results
        return results

    def _get_embeddings(self, texts: List[str], pooling_method: str = 'average'):
        """
        Get embeddings for given texts
        :param texts: list of texts to encode
        :param batch_size:
        :param pooling_method: 'average' or 'cls' are available by default
        :return: np.ndarray with embeddings
        """

        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Processing Batches"):
            batch_texts = texts[i:i + self.batch_size]
            batch_dict = self.tokenizer(batch_texts, max_length=self.max_len, padding=True, truncation=True,
                                        return_tensors='pt')
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            with torch.no_grad():
                outputs = self.model(**batch_dict)
            if pooling_method == 'average':
                batch_embeddings = self._average_pool(outputs, batch_dict['attention_mask'])
            elif pooling_method == 'cls':
                batch_embeddings = self._cls_pool(outputs)
            elif pooling_method == 'pooler':
                batch_embeddings = self._pooler_pool(outputs)
            elif pooling_method == 'last_token':
                batch_embeddings = self._last_token_pool(outputs, batch_dict['attention_mask'])
            else:
                raise ValueError(f"Unknown pooling method: {pooling_method}")

            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def _average_pool(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden_states = model_output.last_hidden_state
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _cls_pool(self, model_output: torch.Tensor) -> torch.Tensor:
        return model_output.last_hidden_state[:, 0, :]

    def _pooler_pool(self, model_output: torch.Tensor) -> torch.Tensor:
        if not hasattr(model_output, "pooler_output") or model_output.pooler_output is None:
            raise ValueError("The selected model does not provide pooler_output.")
        return model_output.pooler_output

    def _last_token_pool(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden_states = model_output.last_hidden_state
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
