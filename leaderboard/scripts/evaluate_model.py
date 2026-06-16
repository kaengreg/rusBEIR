import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from rusBeIR.beir.datasets.data_loader_hf import HFDataLoader
from rusBeIR.beir.retrieval.evaluation import EvaluateRetrieval
from rusBeIR.beir.retrieval.search.base import BaseSearch


DATASETS_PATH = REPO_ROOT/"leaderboard"/ "data"/"datasets.json"
DEFAULT_OUTPUT = REPO_ROOT/"leaderboard"/"data"/"results.jsonl"


def load_datasets() -> list[dict[str, Any]]:
    with DATASETS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def safe_name(value: str) -> str:
    return value.replace("/", "__").replace(":", "_")


def raw_results_path(results_dir: Path, model_name: str, dataset: dict[str, Any]) -> Path:
    return results_dir/f"results_{model_name}_{dataset['name']}_{dataset['split']}.json"


def merge_metrics(*metric_groups: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for group in metric_groups:
        merged.update(group)
    return merged


def average_metrics(per_dataset: dict[str, dict[str, float]]) -> dict[str, float]:
    values_by_metric: dict[str, list[float]] = defaultdict(list)
    for metrics in per_dataset.values():
        for key, value in metrics.items():
            values_by_metric[key].append(float(value))
    return {
        key: round(sum(values) / len(values), 5)
        for key, values in sorted(values_by_metric.items())
        if values
    }


def retrieve(model, corpus: dict[str, dict[str, str]], queries: dict[str, str], top_k: int,
             query_batch_size: int, num_workers: int) -> dict[str, dict[str, float]]:
    if isinstance(model, BaseSearch):
        return EvaluateRetrieval(retriever=model, k_values=[top_k]).retrieve(corpus, queries)
    return model.retrieve(queries, corpus, top_n=top_k, data_batch_size=query_batch_size, num_workers=num_workers)


def build_dense_model(args: argparse.Namespace, device: str):
    from rusBeIR.retrieval.models.dense.DenseHFModels import DenseHFModels

    class ConfigurableDenseModel(DenseHFModels):
        def __init__(self, model_name: str, maxlen: int, batch_size: int, device: str, pooling_method: str,
                     query_prefix: str, passage_prefix: str, model_sep: str, padding_side: str | None,
                     model_loader: str):
            self.model_loader = model_loader
            super().__init__(model_name=model_name, maxlen=maxlen, batch_size=batch_size, device=device,
                             model_sep=model_sep, padding_side=padding_side)
            self.pooling_method = pooling_method
            self.query_prefix = query_prefix
            self.passage_prefix = passage_prefix

        def load_model(self, model_name: str, device: str = "cuda"):
            if self.model_loader == "t5-encoder":
                from transformers import AutoTokenizer, T5EncoderModel

                model = T5EncoderModel.from_pretrained(model_name).to(device)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model.eval()
                return model, tokenizer
            return super().load_model(model_name, device)

        def encode_queries(self, queries, pooling_method=None, prefix=None):
            return super().encode_queries(queries,
                pooling_method=pooling_method or self.pooling_method,
                prefix=self.query_prefix if prefix is None else prefix)

        def encode_corpus(self, corpus, pooling_method=None, prefix=None):
            return super().encode_corpus(corpus,
                pooling_method=pooling_method or self.pooling_method,
                prefix=self.passage_prefix if prefix is None else prefix)

    return ConfigurableDenseModel(
        model_name=args.model_id,
        maxlen=args.maxlen,
        batch_size=args.batch_size,
        device=device,
        pooling_method=args.pooling,
        query_prefix=args.query_prefix,
        passage_prefix=args.passage_prefix,
        model_sep=args.model_sep,
        padding_side=args.padding_side,
        model_loader=args.model_loader,
    )


def build_model(args: argparse.Namespace):
    if args.model_type == "dense":
        device = select_device(args.device)
        return build_dense_model(args, device), device

    if args.model_type == "reranker":
        from rusBeIR.beir.reranking import Rerank
        from rusBeIR.beir.reranking.models import CrossEncoder

        device = select_device(args.device)
        cross_encoder = CrossEncoder(args.model_id, device=device)
        return Rerank(cross_encoder, batch_size=args.rerank_batch_size, max_length=args.rerank_max_length), device

    if args.sparse_model == "bm25s":
        from rusBeIR.retrieval.models.sparse.bm25s import BM25s

        return BM25s(method=args.bm25_method, k1=args.bm25_k1, b=args.bm25_b), "cpu"

    if args.sparse_model == "tfidf":
        from rusBeIR.retrieval.models.sparse.tfidf import TfidfSearch

        return TfidfSearch(
            lowercase=not args.no_lowercase,
            ngram_range=(args.tfidf_min_ngram, args.tfidf_max_ngram),
            max_features=args.tfidf_max_features,
        ), "cpu"
    
    raise SystemExit(f"Unknown sparse model: {args.sparse_model}")


def evaluate_dataset(model, model_name: str, dataset: dict[str, Any], k_values: list[int], text_type: str,
                     query_batch_size: int, num_workers: int, raw_results_dir: Path | None,
                     model_type: str, first_stage_results_dir: Path | None,
                     first_stage_model_name: str | None, rerank_top_k: int | None) -> dict[str, float]:
    
    corpus, queries, qrels = HFDataLoader(hf_repo=dataset["hf_repo"], hf_repo_qrels=dataset["qrels_repo"],
        streaming=False, keep_in_memory=False, text_type=text_type).load(split=dataset["split"])

    if model_type == "reranker":
        if first_stage_results_dir is None or first_stage_model_name is None:
            raise ValueError("Reranker evaluation requires first-stage results.")

        first_stage_path = raw_results_path(first_stage_results_dir, first_stage_model_name, dataset)
        if not first_stage_path.exists():
            raise FileNotFoundError(f"First-stage results not found: {first_stage_path}")

        with first_stage_path.open("r", encoding="utf-8") as file:
            first_stage_results = json.load(file)

        results = model.rerank(corpus, queries, first_stage_results, top_k=rerank_top_k or max(k_values))
    else:
        results = retrieve(model=model, corpus=corpus, queries=queries, top_k=max(k_values), query_batch_size=query_batch_size, num_workers=num_workers)

    if raw_results_dir is not None:
        raw_results_dir.mkdir(parents=True, exist_ok=True)
        output_path = raw_results_path(raw_results_dir, model_name, dataset)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False)

    retriever = EvaluateRetrieval(k_values=k_values)
    ndcg, map_scores, recall, precision = retriever.evaluate(qrels=qrels, results=results, k_values=k_values)
    mrr = retriever.evaluate_custom(qrels, results, k_values, "mrr")
    return merge_metrics(ndcg, map_scores, recall, precision, mrr)

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return[]
    
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_n, line in enumerate(file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} at line {line_n}: {error}")
    return records


def find_existing_record(path: Path, model_id: str) -> dict[str, Any] | None:
    for record in reversed(read_jsonl(path)):
        if record.get("model_id") == model_id:
            return record
    return None


def insert_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [item for item in read_jsonl(path) if item.get("model_id") != record["model_id"]]
    records.append(record)

    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        for item in records:
            file.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
            file.write("\n")
    tmp_path.replace(path)


def build_result_record(args, record_model_id: str, model_name: str, organization: str,
                        hardware: str, started: float, per_dataset: dict[str, dict[str, float]]) -> dict[str, Any]:
    record = {
        "model_id": record_model_id,
        "model_name": model_name,
        "organization": organization,
        "type": args.model_type,
        "date": date.today().isoformat(),
        "verified": False,
        "hardware": hardware,
        "runtime_seconds": round(time.time() - started, 2),
        "source_url": args.source_url,
        "scores": {
            "average": average_metrics(per_dataset),
            "datasets": per_dataset,
        },
        "notes": args.notes,
    }
    if args.model_type == "reranker":
        record["base_model_id"] = args.first_stage_model_id
        record["rerank_top_k"] = args.rerank_top_k or max(args.k_values)
    return record


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="", help="Hugging Face model id for dense and reranker models. Defaults to sparse model name for sparse runs.")
    parser.add_argument("--model-name", default="", help="Display name. Defaults to the last segment of --model-id.")
    parser.add_argument("--model-type", default="dense", choices=["dense", "sparse", "reranker"])
    parser.add_argument("--sparse-model", default="bm25s", choices=["bm25s", "tfidf"], help="Sparse retriever to use when --model-type sparse.")
    parser.add_argument("--first-stage-model-id", "--base-model-id", dest="first_stage_model_id", default="",
                        help="Model id/name whose raw retrieval results will be reranked.")
    parser.add_argument("--first-stage-results-dir", "--base-results-dir", dest="first_stage_results_dir", type=Path,
                        default=None, help="Directory with first-stage raw result JSON files.")
    parser.add_argument("--rerank-top-k", type=int, default=None, help="Number of first-stage hits to rerank. Defaults to max(k-values).")
    parser.add_argument("--rerank-batch-size", type=int, default=32)
    parser.add_argument("--rerank-max-length", type=int, default=512)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSONL file to update with the leaderboard row.")
    parser.add_argument("--raw-results-dir", type=Path, default=None, help="Optional directory for raw retrieval results.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset names to evaluate. Defaults to all official datasets.")
    parser.add_argument("--limit-datasets", type=int, default=None, help="Evaluate only the first N selected datasets.")
    parser.add_argument("--text-type", choices=["processed_text", "text"], default="text")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10, 100])
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--maxlen", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128, help="Tokenizer/model batch size.")
    parser.add_argument("--query-batch-size", type=int, default=16, help="Query scoring batch size.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pooling", choices=["average", "cls", "pooler", "last_token"], default="average")
    parser.add_argument("--model-loader", choices=["auto", "t5-encoder"], default="auto",
                        help="Model loader for dense encoders. Use t5-encoder for encoder-only T5 embedding models such as FRIDA.")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--passage-prefix", default="")
    parser.add_argument("--model-sep", default="[SEP]")
    parser.add_argument("--padding-side", choices=["left", "right"], default=None,
                        help="Tokenizer padding side. Qwen3 Embedding models should use left padding with last_token pooling.")
    parser.add_argument("--bm25-method", default="lucene")
    parser.add_argument("--bm25-k1", type=float, default=None)
    parser.add_argument("--bm25-b", type=float, default=None)
    parser.add_argument("--tfidf-min-ngram", type=int, default=1)
    parser.add_argument("--tfidf-max-ngram", type=int, default=1)
    parser.add_argument("--tfidf-max-features", type=int, default=None)
    parser.add_argument("--no-lowercase", action="store_true")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse existing per-dataset scores for this model_id from --output and skip already computed datasets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    requested = set(args.datasets or [])
    datasets = [dataset
        for dataset in load_datasets()
        if dataset.get("official", True) and (not requested or dataset["name"] in requested)]
    
    if args.limit_datasets is not None:
        datasets = datasets[: args.limit_datasets]

    if not datasets:
        raise SystemExit("No datasets selected.")

    if args.model_type in {"dense", "reranker"} and not args.model_id:
        raise SystemExit("--model-id is required for dense and reranker models.")

    if args.model_type == "sparse" and not args.model_id:
        args.model_id = args.sparse_model

    first_stage_model_name = None
    if args.model_type == "reranker":
        if not args.first_stage_model_id:
            raise SystemExit("--first-stage-model-id is required for reranker evaluation.")
        if args.first_stage_results_dir is None:
            raise SystemExit("--first-stage-results-dir is required for reranker evaluation.")
        first_stage_model_name = safe_name(args.first_stage_model_id)
        missing_results = [
            raw_results_path(args.first_stage_results_dir, first_stage_model_name, dataset)
            for dataset in datasets
            if not raw_results_path(args.first_stage_results_dir, first_stage_model_name, dataset).exists()
        ]
        if missing_results:
            missing = "\n".join(str(path) for path in missing_results)
            raise SystemExit(f"Missing first-stage result files:\n{missing}")

    record_model_id = args.model_id
    if args.model_type == "reranker":
        record_model_id = f"{args.first_stage_model_id}+{args.model_id}"

    model_name = args.model_name or record_model_id.split("/")[-1]
    organization = args.model_id.split("/", 1)[0] if "/" in args.model_id else ""

    model, hardware = build_model(args)

    per_dataset: dict[str, dict[str, float]] = {}
    if args.resume:
        existing_record = find_existing_record(args.output, record_model_id)
        existing_scores = (existing_record or {}).get("scores", {}).get("datasets", {})
        if isinstance(existing_scores, dict):
            per_dataset.update(existing_scores)

    model_file_name = safe_name(args.model_id)
    if args.model_type == "reranker":
        model_file_name = f"{model_file_name}__rerank_{first_stage_model_name}"

    for dataset in datasets:
        dataset_name = dataset["name"]
        if args.resume and dataset_name in per_dataset:
            print(f"Skipping {dataset_name} ({dataset['split']}): already present in {args.output}", flush=True)
            continue

        print(f"Evaluating {dataset_name} ({dataset['split']})", flush=True)
        per_dataset[dataset_name] = evaluate_dataset(
            model=model,
            model_name=model_file_name,
            dataset=dataset,
            k_values=args.k_values,
            text_type=args.text_type,
            query_batch_size=args.query_batch_size,
            num_workers=args.num_workers,
            raw_results_dir=args.raw_results_dir,
            model_type=args.model_type,
            first_stage_results_dir=args.first_stage_results_dir,
            first_stage_model_name=first_stage_model_name,
            rerank_top_k=args.rerank_top_k
        )
        checkpoint = build_result_record(args, record_model_id, model_name, organization, hardware, started, per_dataset)
        insert_jsonl_record(args.output, checkpoint)
        print(f"Saved checkpoint for {dataset_name} to {args.output}", flush=True)

    record = build_result_record(args, record_model_id, model_name, organization, hardware, started, per_dataset)
    insert_jsonl_record(args.output, record)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
