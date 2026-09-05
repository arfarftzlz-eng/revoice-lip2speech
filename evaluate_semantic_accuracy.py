"""Evaluate meaning preservation separately from strict transcript WER."""

from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


LABEL_SCORES = {"equivalent": 1.0, "mostly_correct": 0.5, "incorrect": 0.0}


def endpoint(api_base: str) -> str:
    base = api_base.strip().rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("API base must be an absolute HTTP(S) URL")
    return base if parsed.path.endswith("/chat/completions") else base + "/chat/completions"


def judge_batch(rows: list[dict], *, api_key: str, api_base: str, model: str) -> list[dict]:
    system = (
        "You are an independent evaluator of English lip-reading transcripts. "
        "Compare each hypothesis only with its reference meaning. Ignore case, "
        "punctuation, harmless contractions, and grammatical paraphrases. Use "
        "equivalent when the proposition, entities, actions, attributes, negation, "
        "modality, and time are preserved; mostly_correct when the central intent is "
        "clear but one minor detail is missing or changed; incorrect when a key "
        "entity/action/negation/detail changes or the sentence is not understandable. "
        "Judge visual and qwen independently. Return JSON only and do not favor qwen."
    )
    items = [
        {
            "id": f"{row['take']}-{int(row['id']):02d}",
            "reference": row["reference"],
            "visual": row["visual_top1"],
            "qwen": row["qwen_selected_text"],
        }
        for row in rows
    ]
    user = {
        "items": items,
        "required_json": {
            "items": [
                {
                    "id": "1-01",
                    "visual": {"label": "equivalent", "reason": "short reason"},
                    "qwen": {"label": "mostly_correct", "reason": "short reason"},
                }
            ]
        },
        "allowed_labels": list(LABEL_SCORES),
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "temperature": 0.0,
        "max_tokens": 4000,
        "stream": False,
        "response_format": {"type": "json_object"},
        "enable_thinking": False,
    }
    request = urllib.request.Request(
        endpoint(api_base),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read(500).decode("utf-8", errors="replace").replace(api_key, "[REDACTED]")
        raise RuntimeError(f"Semantic API returned HTTP {error.code}: {detail}") from error
    content = body["choices"][0]["message"]["content"]
    result = json.loads(content)
    judged = result.get("items")
    if not isinstance(judged, list):
        raise RuntimeError("Semantic API returned no items array")
    by_id = {str(item.get("id")): item for item in judged if isinstance(item, dict)}
    expected_ids = {item["id"] for item in items}
    if set(by_id) != expected_ids:
        raise RuntimeError("Semantic API did not return every requested item exactly once")
    for item in by_id.values():
        for field in ("visual", "qwen"):
            label = str(item.get(field, {}).get("label", ""))
            if label not in LABEL_SCORES:
                raise RuntimeError(f"Semantic API returned invalid label: {label}")
    return [by_id[item["id"]] for item in items]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows: list[dict], prefix: str) -> dict:
    labels = [row[f"{prefix}_semantic_label"] for row in rows]
    scores = [float(row[f"{prefix}_semantic_score"]) for row in rows]
    counts = {label: labels.count(label) for label in LABEL_SCORES}
    total = len(rows)
    return {
        "clips": total,
        "equivalent": counts["equivalent"],
        "mostly_correct": counts["mostly_correct"],
        "incorrect": counts["incorrect"],
        "meaning_exact_rate": counts["equivalent"] / total if total else None,
        "meaning_acceptable_rate": (
            (counts["equivalent"] + counts["mostly_correct"]) / total if total else None
        ),
        "weighted_semantic_accuracy": sum(scores) / total if total else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument(
        "--api-base", default="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    parser.add_argument("--model", default="qwen3.7-flash")
    args = parser.parse_args()
    key = os.environ.get(args.api_key_env, "").strip()
    if not key:
        raise RuntimeError(f"{args.api_key_env} is empty")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    summary_path = (args.summary or output_path.with_suffix(".summary.json")).resolve()
    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        source = list(csv.DictReader(file))

    output_rows = []
    for start in range(0, len(source), args.batch_size):
        batch = source[start : start + args.batch_size]
        print(f"Judging {start + 1}-{start + len(batch)} / {len(source)}", flush=True)
        judgments = judge_batch(
            batch, api_key=key, api_base=args.api_base, model=args.model
        )
        for row, judgment in zip(batch, judgments):
            enriched = dict(row)
            for field in ("visual", "qwen"):
                label = judgment[field]["label"]
                enriched[f"{field}_semantic_label"] = label
                enriched[f"{field}_semantic_score"] = LABEL_SCORES[label]
                enriched[f"{field}_semantic_reason"] = str(
                    judgment[field].get("reason", "")
                )
            output_rows.append(enriched)
        write_csv(output_path, output_rows)

    report = {
        "judge_model": args.model,
        "label_scores": LABEL_SCORES,
        "visual_top1": metrics(output_rows, "visual"),
        "qwen_accuracy_first": metrics(output_rows, "qwen"),
        "note": (
            "LLM-assisted semantic evaluation; equivalent ignores harmless "
            "contractions and paraphrases. Human review is recommended for publication."
        ),
    }
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
