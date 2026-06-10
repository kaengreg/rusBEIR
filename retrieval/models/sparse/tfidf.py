from sklearn.feature_extraction.text import TfidfVectorizer
from rusBeIR.beir.retrieval.search.base import BaseSearch
import numpy as np

class TfidfSearch(BaseSearch):
    def __init__(self, lowercase: bool = True, ngram_range: tuple[int, int] = (1, 1), max_features: int | None = None):
        self.vectorizer = TfidfVectorizer(lowercase=lowercase, ngram_range=ngram_range, max_features=max_features)

    @staticmethod
    def doc_text(doc: dict[str, str]) -> str:
        title = doc.get("title") or ""
        text = doc.get("text") or ""
        return f"{title} {text}".strip()

    def search(self, corpus: dict[str, dict[str, str]], queries: dict[str, str], top_k: int,
               score_function: str | None = None, *args, **kwargs) -> dict[str, dict[str, float]]:
        doc_ids = list(corpus.keys())
        query_ids = list(queries.keys())
        documents = [self.doc_text(corpus[doc_id]) for doc_id in doc_ids]
        query_texts = [queries[query_id] for query_id in query_ids]

        doc_matrix = self.vectorizer.fit_transform(documents)
        query_matrix = self.vectorizer.transform(query_texts)
        score_matrix = query_matrix @ doc_matrix.T

        results: dict[str, dict[str, float]] = {}
        for row_id, query_id in enumerate(query_ids):
            scores = score_matrix.getrow(row_id).toarray().ravel()
            candidate_count = min(top_k + 1, len(scores))
            if candidate_count == len(scores):
                top_indices = np.argsort(scores)[::-1]
            else:
                top_indices = np.argpartition(scores, -candidate_count)[-candidate_count:]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            hits: dict[str, float] = {}
            for doc_index in top_indices:
                doc_id = doc_ids[int(doc_index)]
                if doc_id == query_id:
                    continue
                hits[doc_id] = float(scores[doc_index])
                if len(hits) == top_k:
                    break
            results[query_id] = hits
        return results
