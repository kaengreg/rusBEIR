---
title: RusBEIR Leaderboard
emoji: 🔎
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 6.17.3
python_version: 3.11
app_file: app.py
pinned: false
---

# RusBEIR Leaderboard

This directory contains a Hugging Face Space for the RusBEIR benchmark leaderboard.

The Space is intentionally backed by plain files:

- `data/datasets.json` describes the official RusBEIR evaluation tasks.
- `data/results.jsonl` stores reviewed leaderboard rows.
1- `scripts/evaluate_model.py` evaluates a Hugging Face embedding model and updates a result row after every dataset.
- `scripts/export_eval_results.py` exports a row to Hugging Face `.eval_results/*.yaml` format.
- `hub/eval.yaml` is a draft benchmark dataset configuration for Hugging Face Eval Results.

## Local Run

```bash
cd rusBeIR/leaderboard
pip install -r requirements.txt
python app.py
```

Use `evaluation-requirements.txt` for local or GPU-runner model evaluation.

## Evaluate A Model

From the repository root:

```bash
pip install -r rusBeIR/leaderboard/evaluation-requirements.txt
python rusBeIR/leaderboard/scripts/evaluate_model.py \
  --model-id intfloat/multilingual-e5-large \
  --dense-backend faiss \
  --faiss-device auto \
  --faiss-index-dir rusBeIR/leaderboard/faiss_indexes \
  --device cuda \
  --query-prefix "query: " \
  --passage-prefix "passage: " \
  --raw-results-dir rusBeIR/leaderboard/raw/intfloat__multilingual-e5-large \
  --resume
```

Dense retrieval supports two backends:

- `--dense-backend faiss` uses FAISS `IndexFlatIP` over L2-normalized embeddings. This is the recommended backend for
  full benchmark runs and large corpora such as `rus-mmarco` and `rus-miracl`.
- `--dense-backend exact` uses the previous in-memory cosine-similarity implementation. Use it only for debugging or
  small smoke tests.

FAISS can run on CPU or GPU:

```bash
--faiss-device auto  # use GPU when the installed FAISS build supports it, otherwise CPU
--faiss-device cpu   # force CPU FAISS
--faiss-device cuda --faiss-gpu-id 0
```

FAISS indexes are saved in CPU format under `--faiss-index-dir` and reused on repeated runs. Use
`--rebuild-faiss-index` to rebuild an existing index.

`evaluate_model.py` checkpoints progress after every dataset by updating the same JSONL row. If a run is interrupted,
rerun the same command with `--resume` to skip datasets already present in the output row.

For a quick smoke test on one dataset:

```bash
python rusBeIR/leaderboard/scripts/evaluate_model.py \
  --model-id intfloat/multilingual-e5-large \
  --device cuda \
  --datasets rus-scifact
```

Sparse retrievers are evaluated with the same command and result format:

```bash
python rusBeIR/leaderboard/scripts/evaluate_model.py \
  --model-type sparse \
  --sparse-model bm25s \
  --model-name BM25s \
  --datasets rus-scifact
```

```bash
python rusBeIR/leaderboard/scripts/evaluate_model.py \
  --model-type sparse \
  --sparse-model tfidf \
  --model-name TF-IDF \
  --datasets rus-scifact
```

Rerankers are evaluated on top of saved first-stage retrieval results. First save raw results for a base
retriever:

```bash
python rusBeIR/leaderboard/scripts/evaluate_model.py \
  --model-type sparse \
  --sparse-model bm25s \
  --model-name BM25s \
  --raw-results-dir rusBeIR/leaderboard/raw \
  --datasets rus-scifact
```

Then rerank those results:

```bash
python rusBeIR/leaderboard/scripts/evaluate_model.py \
  --model-type reranker \
  --model-id BAAI/bge-reranker-v2-m3 \
  --model-name bge-reranker-v2-m3-over-BM25s \
  --first-stage-model-id bm25s \
  --first-stage-results-dir rusBeIR/leaderboard/raw \
  --raw-results-dir rusBeIR/leaderboard/raw \
  --rerank-top-k 100 \
  --device cuda \
  --datasets rus-scifact
```

The script updates one JSONL row in `data/results.jsonl` for the evaluated `model_id`. Set `"verified": true` only
after the raw retrieval results, command, model revision, and hardware are auditable. The row is rewritten after each
dataset so long runs can be resumed safely.

## Export HF Eval Results

Hugging Face supports decentralized model evaluation metadata through `.eval_results/*.yaml`.
After a model has a JSONL result row, export it with:

```bash
python rusBeIR/leaderboard/scripts/export_eval_results.py \
  --model-id intfloat/multilingual-e5-large \
  --output .eval_results/rusbeir.yaml \
  --include-average
```

The benchmark dataset repo should contain `hub/eval.yaml` at its root before those results can be aggregated
by Hugging Face Eval Results.
