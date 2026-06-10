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
- `scripts/evaluate_model.py` evaluates a Hugging Face embedding model and appends a result row.
- `scripts/export_eval_results.py` exports a row to Hugging Face `.eval_results/*.yaml` format.
- `hub/eval.yaml` is a draft benchmark dataset configuration for Hugging Face Eval Results.

## Local Run

```bash
cd rusBeIR/leaderboard
pip install -r requirements.txt
python app.py
```

`requirements.txt` is intentionally small because the Space only renders reviewed results.
Use `evaluation-requirements.txt` for local or GPU-runner model evaluation.

## Evaluate A Model

From the repository root:

```bash
pip install -r rusBeIR/leaderboard/evaluation-requirements.txt
python rusBeIR/leaderboard/scripts/evaluate_model.py \
  --model-id intfloat/multilingual-e5-large \
  --device cuda \
  --query-prefix "query: " \
  --passage-prefix "passage: " \
  --raw-results-dir rusBeIR/leaderboard/raw/intfloat__multilingual-e5-large
```

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

The script appends a JSONL row to `data/results.jsonl`. Set `"verified": true` only after the raw retrieval
results, command, model revision, and hardware are auditable.

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
