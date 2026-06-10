import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import yaml


DEFAULT_RESULTS = Path(__file__).resolve().parents[1]/"data"/"results.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                records.append(json.loads(line))
    return records


def find_record(records: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    matches = [record for record in records if record.get("model_id") == model_id]
    if not matches:
        raise SystemExit(f"No results found for model_id={model_id!r}")
    return matches[-1]


def eval_result_entry(benchmark_dataset: str, task_id: str, value: float, metric: str, run_date: str, source_url: str, notes: str) -> dict[str, Any]:
    entry = {
        "dataset": {
            "id": benchmark_dataset,
            "task_id": task_id,
        },
        "value": value,
        "date": run_date,
        "notes": f"{metric}; {notes}".strip("; "),
    }
    if source_url:
        entry["source"] = {"url": source_url, "name": "RusBEIR evaluation"}
    return entry


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", default="NDCG@10")
    parser.add_argument("--benchmark-dataset", default="")
    parser.add_argument("--include-average", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    record = find_record(read_jsonl(args.results), args.model_id)
    scores = record.get("scores", {})
    run_date = record.get("date") or date.today().isoformat()
    source_url = record.get("source_url", "")
    notes = record.get("notes", "")

    entries = []
    if args.include_average:
        average_value = scores.get("average", {}).get(args.metric)
        if average_value is not None:
            entries.append(eval_result_entry(args.benchmark_dataset, "average", float(average_value), args.metric, 
                                             run_date, source_url, notes))

    for dataset_name, metrics in sorted(scores.get("datasets", {}).items()):
        if args.metric not in metrics:
            continue
        entries.append(eval_result_entry(args.benchmark_dataset, dataset_name, float(metrics[args.metric]),
                                         args.metric, run_date, source_url, notes))

    if not entries:
        raise SystemExit(f"No metric {args.metric!r} found for model_id={args.model_id!r}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        yaml.safe_dump(entries, file, sort_keys=False, allow_unicode=True)
    print(f"Wrote {len(entries)} eval result entries to {args.output}")


if __name__ == "__main__":
    main()
