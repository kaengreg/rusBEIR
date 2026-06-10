import base64
import inspect
import json
import os
from html import escape
from datetime import date
from pathlib import Path
from typing import Any

os.environ.setdefault("GRADIO_SSR_MODE", "false")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import gradio as gr
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASETS_PATH = ROOT/"data"/"datasets.json"
LOGO_PATH = ROOT/"assets"/"rusBEIR_logo.png"

RESULTS_PATH = Path(os.getenv("RUSBEIR_RESULTS_PATH", ROOT/"data"/"results.jsonl"))
DEFAULT_METRIC = "NDCG@10"
METRICS = ["NDCG@10", "MAP@10", "Recall@10", "P@10", "MRR@10"]
STATIC_COLUMNS = ["Rank", "Model"]
META_COLUMNS = ["Model ID", "Organization", "Type", "Verified", "Date", "Source URL", ]
TRAILING_COLUMNS = META_COLUMNS
DISPLAY_COLUMN_NAMES = {
    "Model ID": "Model\nID",
    "Organization": "Org.",
    "Source URL": "Source\nURL",
    "sberquad-retrieval": "sberquad\nretrieval",
    "ruscibench-retrieval": "ruscibench\nretrieval",
    "wikifacts-articles": "wikifacts\narticles",
    "wikifacts-para": "wikifacts\npara",
    "wikifacts-sents": "wikifacts\nsents",
    "wikifacts-window_2": "wikifacts\nwindow 2",
    "wikifacts-window_3": "wikifacts\nwindow 3",
    "wikifacts-window_4": "wikifacts\nwindow 4",
    "wikifacts-window_5": "wikifacts\nwindow 5",
    "wikifacts-window_6": "wikifacts\nwindow 6",
}
CUSTOM_CSS = """
:root {
  --rusbeir-bg: #f7f8fb;
  --rusbeir-card: #ffffff;
  --rusbeir-text: #111827;
  --rusbeir-muted: #64748b;
  --rusbeir-line: #e2e8f0;
  --rusbeir-accent: #b45309;
  --rusbeir-accent-soft: #fff7ed;
  --rusbeir-green: #047857;
  --rusbeir-soft: #f8fafc;
  --rusbeir-table-head: #f8fafc;
  --rusbeir-table-alt: #fcfcfd;
  --rusbeir-table-border: #edf2f7;
  --rusbeir-shadow: rgba(15, 23, 42, 0.06);
  --rusbeir-panel-shadow: rgba(15, 23, 42, 0.04);
}

@media (prefers-color-scheme: dark) {
  :root {
    --rusbeir-bg: #0f1117;
    --rusbeir-card: #1f2028;
    --rusbeir-text: #f3f4f6;
    --rusbeir-muted: #c1c7d0;
    --rusbeir-line: #3f424c;
    --rusbeir-accent: #f59e0b;
    --rusbeir-accent-soft: #322719;
    --rusbeir-green: #34d399;
    --rusbeir-soft: #272933;
    --rusbeir-table-head: #272933;
    --rusbeir-table-alt: #23252e;
    --rusbeir-table-border: #383b46;
    --rusbeir-shadow: rgba(0, 0, 0, 0.25);
    --rusbeir-panel-shadow: rgba(0, 0, 0, 0.18);
  }
}

.dark,
body.dark,
[data-theme="dark"] {
  --rusbeir-bg: #0f1117;
  --rusbeir-card: #1f2028;
  --rusbeir-text: #f3f4f6;
  --rusbeir-muted: #c1c7d0;
  --rusbeir-line: #3f424c;
  --rusbeir-accent: #f59e0b;
  --rusbeir-accent-soft: #322719;
  --rusbeir-green: #34d399;
  --rusbeir-soft: #272933;
  --rusbeir-table-head: #272933;
  --rusbeir-table-alt: #23252e;
  --rusbeir-table-border: #383b46;
  --rusbeir-shadow: rgba(0, 0, 0, 0.25);
  --rusbeir-panel-shadow: rgba(0, 0, 0, 0.18);
}

body,
.gradio-container {
  background: var(--rusbeir-bg) !important;
  color: var(--rusbeir-text) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

html,
body {
  overflow-x: hidden !important;
}

.gradio-container {
  width: 100% !important;
  max-width: 1440px !important;
  min-width: 0 !important;
  margin: 0 auto !important;
  padding: 22px !important;
  box-sizing: border-box !important;
}

.rusbeir-shell {
  display: flex;
  flex-direction: column;
  gap: 18px;
  min-width: 0;
  width: 100%;
}

.rusbeir-hero {
  background: var(--rusbeir-card);
  border: 1px solid var(--rusbeir-line);
  border-radius: 18px;
  padding: 24px;
  box-shadow: 0 12px 32px var(--rusbeir-shadow);
  min-width: 0;
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 18px;
  align-items: start;
}

.rusbeir-hero-copy {
  min-width: 0;
}

.rusbeir-logo {
  width: 148px;
  max-width: 24vw;
  height: auto;
  object-fit: contain;
}

.rusbeir-kicker {
  color: var(--rusbeir-accent);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 8px;
}

.rusbeir-title {
  font-size: 42px;
  line-height: 1.05;
  font-weight: 850;
  letter-spacing: 0;
  margin: 0 0 10px;
  overflow-wrap: anywhere;
}

.rusbeir-subtitle {
  color: var(--rusbeir-muted);
  font-size: 16px;
  line-height: 1.55;
  max-width: 860px;
  margin: 0;
  overflow-wrap: anywhere;
}

.rusbeir-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 18px;
}

.rusbeir-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid var(--rusbeir-line);
  border-radius: 999px;
  background: var(--rusbeir-soft);
  color: var(--rusbeir-text);
  padding: 6px 10px;
  font-size: 13px;
  font-weight: 650;
}

.rusbeir-cards {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.rusbeir-card {
  background: var(--rusbeir-card);
  border: 1px solid var(--rusbeir-line);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 8px 24px var(--rusbeir-panel-shadow);
  min-width: 0;
}

.rusbeir-card-label {
  color: var(--rusbeir-muted);
  font-size: 12px;
  font-weight: 750;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.rusbeir-card-value {
  color: var(--rusbeir-text);
  font-size: 28px;
  line-height: 1.15;
  font-weight: 820;
  margin-top: 8px;
  overflow-wrap: anywhere;
}

.rusbeir-card-note {
  color: var(--rusbeir-muted);
  font-size: 13px;
  margin-top: 6px;
}

.rusbeir-panel {
  background: var(--rusbeir-card);
  border: 1px solid var(--rusbeir-line);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 8px 24px var(--rusbeir-panel-shadow);
  min-width: 0;
  overflow-x: visible;
}

.rusbeir-panel > *,
.rusbeir-panel .block,
.rusbeir-panel .wrap,
.rusbeir-panel .form,
.rusbeir-panel .container {
  min-width: 0 !important;
}

.rusbeir-filters,
.rusbeir-filters .block,
.rusbeir-filters .form,
.rusbeir-filters .wrap,
.rusbeir-filters .container,
.rusbeir-filters .input-container,
.rusbeir-filters .input-wrapper,
.rusbeir-filters .secondary-wrap,
.rusbeir-filters fieldset,
.rusbeir-filters label,
.rusbeir-filters input,
.rusbeir-filters textarea,
.rusbeir-filters select,
.rusbeir-filters button {
  border-radius: 0 !important;
}

.rusbeir-verified-filter {
  align-self: stretch !important;
}

.rusbeir-verified-filter .wrap,
.rusbeir-verified-filter .block,
.rusbeir-verified-filter label {
  height: 100% !important;
}

.rusbeir-verified-filter label {
  display: flex !important;
  align-items: center !important;
  padding-top: 30px !important;
  box-sizing: border-box !important;
}

.rusbeir-section-title {
  color: var(--rusbeir-text);
  font-size: 18px;
  font-weight: 800;
  margin: 0 0 4px;
}

.rusbeir-section-note {
  color: var(--rusbeir-muted);
  font-size: 13px;
  margin: 0 0 14px;
}

.rusbeir-citation {
  margin: 10px 0 0;
  padding: 14px;
  border: 1px solid var(--rusbeir-line);
  background: var(--rusbeir-soft);
  color: var(--rusbeir-text);
  overflow-x: auto;
  white-space: pre-wrap;
  font-size: 12px;
  line-height: 1.45;
}

.rusbeir-table-scroll {
  width: 100%;
  max-height: 720px;
  overflow: auto;
  border: 1px solid var(--rusbeir-line);
  border-radius: 14px;
  background: var(--rusbeir-card);
}

.rusbeir-table {
  border-collapse: separate;
  border-spacing: 0;
  border: 0 !important;
  min-width: 100%;
  width: max-content;
  table-layout: fixed;
  font-size: 13px;
}

.rusbeir-table th,
.rusbeir-table td {
  border: 0 !important;
  border-right: 1px solid var(--rusbeir-line) !important;
  border-bottom: 1px solid var(--rusbeir-table-border) !important;
  padding: 10px 10px;
  color: var(--rusbeir-text);
  background: var(--rusbeir-card);
  vertical-align: middle;
  overflow-wrap: anywhere;
}

.rusbeir-table th {
  position: sticky;
  top: 0;
  z-index: 4;
  background: var(--rusbeir-table-head);
  color: var(--rusbeir-text);
  font-weight: 800;
  white-space: normal;
  line-height: 1.15;
  vertical-align: bottom;
  border-bottom: 1px solid var(--rusbeir-line) !important;
}

.rusbeir-sort-button {
  width: 100%;
  min-height: auto !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  color: inherit !important;
  box-shadow: none !important;
  padding: 0 !important;
  font: inherit !important;
  font-weight: inherit !important;
  line-height: inherit !important;
  text-align: inherit !important;
  cursor: pointer;
}

.rusbeir-sort-button:hover {
  color: #b45309 !important;
}

.rusbeir-sort-indicator {
  color: #b45309;
  font-size: 11px;
  font-weight: 800;
}

.rusbeir-table td {
  height: 48px;
  font-weight: 650;
}

.rusbeir-table tr:nth-child(even) td {
  background: var(--rusbeir-table-alt);
}

.rusbeir-table .col-rank {
  width: 58px;
  min-width: 58px;
  max-width: 58px;
  text-align: center;
}

.rusbeir-table .col-model {
  width: 260px;
  min-width: 260px;
  max-width: 260px;
}

.rusbeir-table .col-average {
  width: 104px;
  min-width: 104px;
  max-width: 104px;
  text-align: right;
}

.rusbeir-table .col-meta {
  width: 110px;
  min-width: 110px;
  max-width: 110px;
}

.rusbeir-table .col-model-id {
  width: 260px;
  min-width: 260px;
  max-width: 260px;
}

.rusbeir-table .col-dataset {
  width: 96px;
  min-width: 96px;
  max-width: 96px;
  text-align: right;
}

.rusbeir-table .col-date {
  width: 112px;
  min-width: 112px;
  max-width: 112px;
}

.rusbeir-table .col-source {
  width: 220px;
  min-width: 220px;
  max-width: 220px;
}

.rusbeir-table .sticky-rank,
.rusbeir-table .sticky-model,
.rusbeir-table .sticky-average {
  position: sticky;
  z-index: 3;
}

.rusbeir-table th.sticky-rank,
.rusbeir-table th.sticky-model,
.rusbeir-table th.sticky-average {
  z-index: 6;
}

.rusbeir-table .sticky-rank {
  left: 0;
}

.rusbeir-table .sticky-model {
  left: 58px;
}

.rusbeir-table .sticky-average {
  left: 318px;
  box-shadow: 8px 0 12px var(--rusbeir-shadow);
}

.rusbeir-source-link {
  color: #b45309;
  text-decoration: none;
  font-weight: 700;
}

@media (prefers-color-scheme: dark) {
  .rusbeir-source-link {
    color: #f59e0b;
  }
}

.dark .rusbeir-source-link,
body.dark .rusbeir-source-link,
[data-theme="dark"] .rusbeir-source-link {
  color: #f59e0b;
}

.rusbeir-empty {
  color: var(--rusbeir-muted);
  padding: 18px;
}

button {
  font-weight: 700 !important;
}

.tabs {
  border: 0 !important;
}

@media (max-width: 900px) {
  .gradio-container {
    padding: 12px !important;
  }
  .rusbeir-title {
    font-size: 30px;
  }
  .rusbeir-hero {
    grid-template-columns: 1fr;
  }
  .rusbeir-logo {
    width: 120px;
    max-width: 100%;
  }
  .rusbeir-cards {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 560px) {
  .rusbeir-cards {
    grid-template-columns: 1fr;
  }
}
"""

CUSTOM_JS = """
  window.rusbeirSortTable = function(button) {
    const header = button.closest("th");
    const table = button.closest("table");
    const body = table?.querySelector("tbody");
    if (!header || !table || !body) return;

    const headers = Array.from(header.parentElement.children);
    const index = headers.indexOf(header);
    const sortType = header.dataset.sortType || "text";
    const previousIndex = table.dataset.sortIndex;
    const previousDirection = table.dataset.sortDirection;
    const nextDirection = previousIndex === String(index) && previousDirection === "desc" ? "asc" : "desc";
    table.dataset.sortIndex = String(index);
    table.dataset.sortDirection = nextDirection;

    headers.forEach((item) => {
      const indicator = item.querySelector(".rusbeir-sort-indicator");
      if (indicator) indicator.textContent = "";
    });
    const activeIndicator = header.querySelector(".rusbeir-sort-indicator");
    if (activeIndicator) activeIndicator.textContent = nextDirection === "desc" ? " ▼" : " ▲";

    const rows = Array.from(body.querySelectorAll("tr"));
    rows.sort((left, right) => {
      const leftText = (left.children[index]?.innerText || "").trim();
      const rightText = (right.children[index]?.innerText || "").trim();
      let result;
      if (sortType === "number") {
        const leftNumber = Number.parseFloat(leftText.replace(",", "."));
        const rightNumber = Number.parseFloat(rightText.replace(",", "."));
        const leftValue = Number.isFinite(leftNumber) ? leftNumber : Number.NEGATIVE_INFINITY;
        const rightValue = Number.isFinite(rightNumber) ? rightNumber : Number.NEGATIVE_INFINITY;
        result = leftValue - rightValue;
      } else {
        result = leftText.localeCompare(rightText, undefined, { numeric: true, sensitivity: "base" });
      }
      return nextDirection === "desc" ? -result : result;
    });

    rows.forEach((row, position) => {
      body.appendChild(row);
      const rankCell = row.querySelector(".col-rank");
      if (rankCell) rankCell.textContent = String(position + 1);
    });
  };

  document.addEventListener("click", (event) => {
    const button = event.target.closest(".rusbeir-sort-button");
    if (!button) return;
    event.preventDefault();
    window.rusbeirSortTable(button);
  });
"""

def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def metric_value(metrics: dict[str, Any], metric: str) -> float | None:
    value = metrics.get(metric)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_average(record: dict[str, Any], metric: str, dataset_names: set[str]) -> float | None:
    scores = record.get("scores", {})
    explicit = metric_value(scores.get("average", {}), metric)
    dataset_scores = scores.get("datasets", {})
    covered_dataset_names = {
        dataset_name
        for dataset_name, dataset_metrics in dataset_scores.items()
        if dataset_name in dataset_names and metric_value(dataset_metrics, metric) is not None
    }
    has_full_coverage = bool(dataset_names) and covered_dataset_names == dataset_names

    if explicit is not None and has_full_coverage:
        return explicit

    values = []
    for dataset_name, dataset_metrics in dataset_scores.items():
        if dataset_name not in dataset_names:
            continue
        value = metric_value(dataset_metrics, metric)
        if value is not None:
            values.append(value)
    if not values or len(values) != len(dataset_names):
        return None
    return sum(values) / len(values)


def display_column_name(column: str) -> str:
    return DISPLAY_COLUMN_NAMES.get(column, column)


def format_metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in frame.columns
        if column not in {*STATIC_COLUMNS, *TRAILING_COLUMNS}
    ]
    for column in metric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = values.map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    return frame


def finalize_leaderboard_frame(frame: pd.DataFrame, metric: str, dataset_names: set[str]) -> pd.DataFrame:
    ordered_columns = [
        "Rank",
        "Model",
        metric,
        *sorted(dataset_names),
        *META_COLUMNS,
    ]
    ordered_columns = [column for column in ordered_columns if column in frame.columns]
    frame = frame.loc[:, ordered_columns].copy()
    frame = format_metric_columns(frame)
    return frame.rename(columns={column: display_column_name(column) for column in frame.columns})


def column_class(index: int, column: str, metric: str) -> str:
    if index == 0:
        return "col-rank sticky-rank"
    if index == 1:
        return "col-model sticky-model"
    if column == metric:
        return "col-average sticky-average"
    if column == "Model\nID":
        return "col-model-id"
    if column in {"Org.", "Type", "Verified"}:
        return "col-meta"
    if column == "Date":
        return "col-date"
    if column == "Source\nURL":
        return "col-source"
    return "col-dataset"


def cell_html(value: Any, column: str) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    if column == "Source\nURL" and text:
        return f'<a class="rusbeir-source-link" href="{escape(text, quote=True)}" target="_blank" rel="noopener noreferrer">source</a>'
    return escape(text).replace("\n", "<br>")


def sort_type_for_column(index: int, column: str, metric: str) -> str:
    text_columns = {"Model", "Model\nID", "Org.", "Type", "Verified", "Date", "Source\nURL"}
    if index == 0 or column == metric or column not in text_columns:
        return "number"
    return "text"


def leaderboard_table_html(metric: str, task_filter: str, verified_only: bool, model_filter: str) -> str:
    frame = leaderboard_frame(metric, task_filter, verified_only, model_filter)
    if frame.empty:
        return '<div class="rusbeir-empty">No results match the selected filters.</div>'

    columns = list(frame.columns)
    header = "".join(
        f"""
        <th class="{column_class(index, column, metric)}" data-sort-type="{sort_type_for_column(index, column, metric)}">
          <button class="rusbeir-sort-button" type="button" onclick="event.stopPropagation(); window.rusbeirSortTable && window.rusbeirSortTable(this)">
            {escape(column).replace(chr(10), "<br>")}<span class="rusbeir-sort-indicator"></span>
          </button>
        </th>
        """
        for index, column in enumerate(columns)
    )
    body_rows = []
    for _, row in frame.iterrows():
        cells = "".join(
            f'<td class="{column_class(index, column, metric)}">{cell_html(row[column], column)}</td>'
            for index, column in enumerate(columns)
        )
        body_rows.append(f"<tr>{cells}</tr>")

    return f"""
    <div class="rusbeir-table-scroll">
      <table class="rusbeir-table">
        <thead><tr>{header}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>
    """


def load_datasets() -> list[dict[str, Any]]:
    return read_json(DATASETS_PATH, [])


def load_results() -> list[dict[str, Any]]:
    return read_jsonl(RESULTS_PATH)


def normalize_submission_record(record: dict[str, Any]) -> dict[str, Any]:
    model_id = str(record.get("model_id", "")).strip()
    if not model_id:
        raise ValueError("`model_id` is required.")

    scores = record.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("`scores` must be an object.")

    average_scores = scores.get("average", {})
    dataset_scores = scores.get("datasets", {})
    if average_scores is None:
        average_scores = {}
    if dataset_scores is None:
        dataset_scores = {}
    if not isinstance(average_scores, dict):
        raise ValueError("`scores.average` must be an object.")
    if not isinstance(dataset_scores, dict):
        raise ValueError("`scores.datasets` must be an object.")

    has_metric = any(metric_value(average_scores, metric) is not None for metric in METRICS)
    if not has_metric:
        for metrics in dataset_scores.values():
            if isinstance(metrics, dict) and any(metric_value(metrics, metric) is not None for metric in METRICS):
                has_metric = True
                break
    if not has_metric:
        raise ValueError(f"At least one numeric metric is required: {', '.join(METRICS)}.")

    normalized = dict(record)
    normalized["model_id"] = model_id
    normalized["model_name"] = str(record.get("model_name") or model_id.split("/")[-1]).strip()
    normalized["organization"] = str(record.get("organization") or (model_id.split("/", 1)[0] if "/" in model_id else "")).strip()
    normalized["type"] = str(record.get("type") or "dense").strip()
    normalized["date"] = str(record.get("date") or date.today().isoformat()).strip()
    normalized["verified"] = bool(record.get("verified", False))
    normalized["source_url"] = str(record.get("source_url", "")).strip()
    normalized["scores"] = {"average": average_scores, "datasets": dataset_scores}
    return normalized


def parse_submission_records(record_text: str) -> list[dict[str, Any]]:
    record_text = record_text.strip()
    if not record_text:
        return []

    if record_text.startswith("["):
        parsed = json.loads(record_text)
        if not isinstance(parsed, list):
            raise ValueError("JSON array submission must contain result objects.")
        records = parsed
    else:
        records = []
        for line_no, line in enumerate(record_text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc

    normalized_records = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Submission item {index} must be a JSON object.")
        normalized_records.append(normalize_submission_record(record))
    return normalized_records


def uploaded_file_text(uploaded_file: Any) -> str:
    if uploaded_file is None:
        return ""
    path = uploaded_file
    if isinstance(uploaded_file, dict):
        path = uploaded_file.get("path") or uploaded_file.get("name")
    else:
        path = getattr(uploaded_file, "path", None) or getattr(uploaded_file, "name", None) or uploaded_file
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8-sig")


def append_submission_records(records: list[dict[str, Any]]) -> tuple[int, int]:
    existing = set()
    for item in load_results():
        try:
            existing.add(json.dumps(normalize_submission_record(item), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        except ValueError:
            continue

    serialized_records = []
    skipped = 0
    for record in records:
        serialized = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if serialized in existing:
            skipped += 1
            continue
        existing.add(serialized)
        serialized_records.append(serialized)

    if not serialized_records:
        return 0, skipped

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as file:
        needs_newline = RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size > 0
        if needs_newline:
            with RESULTS_PATH.open("rb") as check_file:
                check_file.seek(-1, os.SEEK_END)
                needs_newline = check_file.read(1) != b"\n"
        if needs_newline:
            file.write("\n")
        file.write("\n".join(serialized_records))

    return len(serialized_records), skipped


def add_submission_record(
    uploaded_file: Any,
    metric: str,
    task_filter: str,
    verified_only: bool,
    model_filter: str,
) -> tuple[str, str, str]:
    try:
        file_text = uploaded_file_text(uploaded_file)
    except OSError as exc:
        return (
            f"Submission was not added: could not read uploaded file: {exc}",
            summary_html(),
            leaderboard_table_html(metric, task_filter, verified_only, model_filter),
        )
    if not file_text.strip():
        return (
            "Upload a non-empty results.jsonl file first.",
            summary_html(),
            leaderboard_table_html(metric, task_filter, verified_only, model_filter),
        )

    try:
        records = parse_submission_records(file_text)
        added, skipped = append_submission_records(records)
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        return (
            f"Submission was not added: {exc}",
            summary_html(),
            leaderboard_table_html(metric, task_filter, verified_only, model_filter),
        )

    if added == 0 and skipped > 0:
        status = f"No new records were added; skipped {skipped} duplicate record(s)."
    else:
        status = f"Added {added} record(s) to `{RESULTS_PATH.name}`."
        if skipped:
            status += f" Skipped {skipped} duplicate record(s)."

    return (
        status,
        summary_html(),
        leaderboard_table_html(metric, task_filter, verified_only, model_filter),
    )


def logo_data_uri() -> str:
    if not LOGO_PATH.exists():
        return ""
    data = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def format_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{numeric_value * 100:.2f}"


def summary_html() -> str:
    datasets = [dataset for dataset in load_datasets() if dataset.get("official", True)]
    records = load_results()
    frame = leaderboard_frame(DEFAULT_METRIC, "All", False, "")
    best_model = "No results yet"
    best_score = "n/a"
    if not frame.empty and DEFAULT_METRIC in frame:
        best_row = frame.iloc[0]
        best_model = str(best_row.get("Model", "No results yet"))
        best_score = format_score(best_row.get(DEFAULT_METRIC))

    types = sorted({str(record.get("type", "")).strip() for record in records if record.get("type")})
    type_text = ", ".join(types) if types else "n/a"
    logo_uri = logo_data_uri()
    logo_html = f'<img class="rusbeir-logo" src="{logo_uri}" alt="rusBEIR logo">' if logo_uri else ""

    return f"""
    <div class="rusbeir-shell">
      <section class="rusbeir-hero">
        <div class="rusbeir-hero-copy">
          <div class="rusbeir-kicker">Russian Information Retrieval Benchmark</div>
          <h1 class="rusbeir-title">rusBEIR Leaderboard</h1>
          <p class="rusbeir-subtitle">
            Compare dense retrievers, sparse baselines, and reranker pipelines on official rusBEIR datasets.
            The default ranking is the macro-average of <strong>{DEFAULT_METRIC}</strong>.
          </p>
          <div class="rusbeir-badges">
            <span class="rusbeir-badge">Metric: {DEFAULT_METRIC}</span>
            <span class="rusbeir-badge">Official datasets: {len(datasets)}</span>
            <span class="rusbeir-badge">Rows: {len(records)}</span>
            <span class="rusbeir-badge">Types: {type_text}</span>
          </div>
        </div>
        {logo_html}
      </section>
      <section class="rusbeir-cards">
        <div class="rusbeir-card">
          <div class="rusbeir-card-label">Best Model</div>
        <div class="rusbeir-card-value">{escape(best_model)}</div>
          <div class="rusbeir-card-note">Highest average {DEFAULT_METRIC}</div>
        </div>
        <div class="rusbeir-card">
          <div class="rusbeir-card-label">Best Score</div>
          <div class="rusbeir-card-value">{best_score}</div>
          <div class="rusbeir-card-note">Shown as percentage points</div>
        </div>
        <div class="rusbeir-card">
          <div class="rusbeir-card-label">Models</div>
          <div class="rusbeir-card-value">{len(records)}</div>
          <div class="rusbeir-card-note">Imported and reviewable JSONL rows</div>
        </div>
        <div class="rusbeir-card">
          <div class="rusbeir-card-label">Datasets</div>
          <div class="rusbeir-card-value">{len(datasets)}</div>
          <div class="rusbeir-card-note">Official benchmark tasks</div>
        </div>
      </section>
    </div>
    """


def leaderboard_frame(metric: str, task_filter: str, verified_only: bool, model_filter: str) -> pd.DataFrame:
    datasets = load_datasets()
    if task_filter != "All":
        datasets = [dataset for dataset in datasets if dataset["task"] == task_filter]
    dataset_names = {dataset["name"] for dataset in datasets if dataset.get("official", True)}

    rows = []
    for record in load_results():
        if verified_only and not record.get("verified", False):
            continue
        model_text = f"{record.get('model_id', '')} {record.get('model_name', '')}".lower()
        if model_filter and model_filter.lower() not in model_text:
            continue

        average = compute_average(record, metric, dataset_names)
        row = {
            "Rank": None,
            "Model": record.get("model_name") or record.get("model_id"),
            "Model ID": record.get("model_id", ""),
            "Organization": record.get("organization", ""),
            "Type": record.get("type", ""),
            metric: average,
            "Verified": "yes" if record.get("verified", False) else "no",
            "Date": record.get("date", ""),
            "Source URL": record.get("source_url", ""),
        }

        dataset_scores = record.get("scores", {}).get("datasets", {})
        for dataset_name in sorted(dataset_names):
            row[dataset_name] = metric_value(dataset_scores.get(dataset_name, {}), metric)
        rows.append(row)

    if not rows:
        frame = pd.DataFrame(columns=[*STATIC_COLUMNS, metric, *TRAILING_COLUMNS])
        return finalize_leaderboard_frame(frame, metric, dataset_names)

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(metric, ascending=False, na_position="last").reset_index(drop=True)
    frame["Rank"] = frame.index + 1
    return finalize_leaderboard_frame(frame, metric, dataset_names)


def datasets_frame() -> pd.DataFrame:
    datasets = load_datasets()
    if not datasets:
        return pd.DataFrame(columns=["Dataset", "Task", "Split", "Corpus repo", "Qrels repo", "Origin"])
    return pd.DataFrame(
        {
            "Dataset": item["name"],
            "Task": item["task"],
            "Split": item["split"],
            "Corpus repo": item["hf_repo"],
            "Qrels repo": item["qrels_repo"],
            "Origin": item["origin"],
        }
        for item in datasets
        if item.get("official", True)
    )


def task_choices() -> list[str]:
    tasks = sorted({dataset["task"] for dataset in load_datasets() if dataset.get("official", True)})
    return ["All", *tasks]


with gr.Blocks(title="rusBEIR Leaderboard") as demo:
    summary = gr.HTML(summary_html())

    with gr.Tabs():
        with gr.Tab("Leaderboard"):
            with gr.Column(elem_classes=["rusbeir-panel"]):
                gr.HTML(
                    """
                    <h2 class="rusbeir-section-title">Model Rankings</h2>
                    <p class="rusbeir-section-note">
                      Filter by task family, model name, or verification status. Scores are stored as fractions;
                      leaderboard rankings use the selected average metric.
                    </p>
                    """
                )
                with gr.Row(elem_classes=["rusbeir-filters"]):
                    metric = gr.Dropdown(METRICS, value=DEFAULT_METRIC, label="Metric", min_width=150)
                    task_filter = gr.Dropdown(task_choices(), value="All", label="Task", min_width=180)
                    model_filter = gr.Textbox(label="Model", placeholder="intfloat, BGE, FRIDA...", min_width=220)
                    verified_only = gr.Checkbox(value=False, label="Verified", min_width=120, elem_classes=["rusbeir-verified-filter"])

                gr.HTML('<div class="rusbeir-section-note">Results</div>')
                table = gr.HTML(value=leaderboard_table_html(DEFAULT_METRIC, "All", False, ""))

                for control in [metric, task_filter, verified_only, model_filter]:
                    control.change(
                        leaderboard_table_html,
                        inputs=[metric, task_filter, verified_only, model_filter],
                        outputs=table,
                    )

                reload_button = gr.Button("Reload results", variant="secondary")
                reload_button.click(
                    leaderboard_table_html,
                    inputs=[metric, task_filter, verified_only, model_filter],
                    outputs=table,
                )

        with gr.Tab("Datasets"):
            with gr.Column(elem_classes=["rusbeir-panel"]):
                gr.HTML(
                    """
                    <h2 class="rusbeir-section-title">Official Datasets</h2>
                    <p class="rusbeir-section-note">
                      rusBEIR tasks used for the default macro-average ranking.
                    </p>
                    """
                )
                gr.Dataframe(value=datasets_frame(), label="Datasets", interactive=False, wrap=True, max_height=720)

        with gr.Tab("Submit"):
            with gr.Column(elem_classes=["rusbeir-panel"]):
                gr.HTML(
                    """
                    <h2 class="rusbeir-section-title">Submit Results</h2>
                    <p class="rusbeir-section-note">
                      Run the evaluator outside the Space and upload the generated <code>results.jsonl</code>.
                      Accepted files contain one JSON result object per line.
                      </br>
                      </br>
                      We strongly recommend checking your results before uploading them to the Leaderboard. 
                      Retracting results is a manual process and can be handled only by @kaengreg.
                    </p>
                    """
                )
                gr.Markdown(
                    """
                    Example:
                    ```bash
                    python leaderboard/scripts/evaluate_model.py --model-id intfloat/multilingual-e5-large --device cuda
                    ```
                    """
                )
                results_file = gr.File(
                    label="Upload results.jsonl",
                    file_types=[".jsonl", ".json"],
                    type="filepath",
                )
                submit_status = gr.Markdown()
                add_button = gr.Button("Add to leaderboard", variant="primary")
                add_button.click(
                    add_submission_record,
                    inputs=[results_file, metric, task_filter, verified_only, model_filter],
                    outputs=[submit_status, summary, table],
                )

        with gr.Tab("About"):
            with gr.Column(elem_classes=["rusbeir-panel"]):
                gr.HTML(
                    """
                    <h2 class="rusbeir-section-title">About rusBEIR</h2>
                    <p class="rusbeir-section-note">
                      rusBEIR is a Russian BEIR-style benchmark for zero-shot information retrieval.
                      The leaderboard is backed by a plain JSONL file, so every row can be reviewed or
                      mirrored to a Hugging Face Dataset.
                    </p>
                    <p class="rusbeir-section-note">
                      Verified rows should point to reproducible logs or a commit with generated retrieval results.
                    </p>
                    <p class="rusbeir-section-note">
                      Project repository:
                      <a class="rusbeir-source-link" href="https://github.com/kaengreg/rusBEIR" target="_blank" rel="noopener noreferrer">kaengreg/rusBEIR</a>
                    </p>
                    <h3 class="rusbeir-section-title">Citation</h3>
                    <pre class="rusbeir-citation"><code>@inproceedings{kovalev2025building,
  title={Building Russian Benchmark for Evaluation of Information Retrieval Models},
  author={Kovalev, Grigory and Tikhomirov, Mikhail and Kozhevnikov, Evgeny and Kornilov, Max and Loukachevitch, Natalia},
  booktitle={Proceedings of the International Conference “Dialogue},
  volume={2025},
  year={2025}
}</code></pre>
                    """
                )


if __name__ == "__main__":
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": int(os.getenv("PORT", "7860")),
        "show_error": True,
        "css": CUSTOM_CSS,
        "js": CUSTOM_JS,
    }
    if "ssr_mode" in inspect.signature(demo.launch).parameters:
        launch_kwargs["ssr_mode"] = False
    demo.launch(**launch_kwargs)
