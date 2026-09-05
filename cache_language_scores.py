"""Score a fixed N-best cache once and persist only non-secret Qwen outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from language_reranker import RecognitionCandidate, request_language_scores


def load_candidate(raw: dict, fallback_rank: int) -> RecognitionCandidate:
    token_ids = tuple(int(value) for value in raw.get("token_ids", []))
    if not token_ids:
        token_ids = tuple(range(max(1, int(raw.get("token_count", 1)))))
    return RecognitionCandidate(
        rank=int(raw.get("rank", fallback_rank)),
        text=str(raw["text"]),
        token_ids=token_ids,
        score=float(raw.get("score", 0.0)),
        component_scores={
            str(name): float(value)
            for name, value in (raw.get("component_scores") or {}).items()
        },
    )


def write_rows(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="CSV produced by evaluate_nbest.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--api-base",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--model", default="qwen3.7-flash")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    key = os.environ.get(args.api_key_env, "").strip()
    if not key:
        raise RuntimeError(f"{args.api_key_env} is empty")

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        source = list(csv.DictReader(file))

    existing: dict[tuple[int, int], dict] = {}
    if args.resume and output_path.is_file():
        with output_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                if not row.get("error"):
                    existing[(int(row["take"]), int(row["id"]))] = row

    rows: list[dict] = []
    for index, sample in enumerate(source, start=1):
        sample_key = (int(sample["take"]), int(sample["id"]))
        if sample_key in existing:
            rows.append(existing[sample_key])
            print(f"[{index:02d}/{len(source):02d}] take={sample_key[0]} id={sample_key[1]:02d} cached")
            continue

        raw_candidates = json.loads(sample["candidates_json"])
        candidates = [
            load_candidate(raw, rank)
            for rank, raw in enumerate(raw_candidates, start=1)
        ]
        print(
            f"[{index:02d}/{len(source):02d}] take={sample_key[0]} "
            f"id={sample_key[1]:02d} candidates={len(candidates)}",
            flush=True,
        )
        try:
            scores, selected_id, confidence = request_language_scores(
                candidates,
                api_key=key,
                api_base=args.api_base,
                model=args.model,
            )
            error = ""
        except Exception as exception:
            scores, selected_id, confidence = {}, "", ""
            error = f"{type(exception).__name__}: {exception}"

        row = {
            "take": sample_key[0],
            "id": sample_key[1],
            "group": sample["group"],
            "filename": sample["filename"],
            "reference": sample["reference"],
            "reference_words": sample["reference_words"],
            "candidates_json": json.dumps(raw_candidates, ensure_ascii=False),
            "language_scores_json": json.dumps(scores, ensure_ascii=False),
            "language_selected_id": selected_id,
            "confidence": confidence,
            "model": args.model,
            "error": error,
        }
        rows.append(row)
        write_rows(rows, output_path)
        print(
            f"  selected={selected_id or '-'} confidence={confidence or '-'} "
            f"error={error or 'none'}",
            flush=True,
        )

    failures = sum(bool(row["error"]) for row in rows)
    print(f"Saved {len(rows)} rows to {output_path}; failures={failures}", flush=True)


if __name__ == "__main__":
    main()
