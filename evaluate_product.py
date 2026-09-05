"""Evaluate the deployed Gradio VSR path against a plain-text manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

def normalize_words(text: str) -> list[str]:
    return re.findall(r"[A-Z0-9]+(?:'[A-Z0-9]+)?", (text or "").upper())


def edit_counts(reference: Iterable[str], hypothesis: Iterable[str]) -> tuple[int, int, int]:
    """Return substitution, deletion and insertion counts."""
    ref = list(reference)
    hyp = list(hypothesis)
    table = [[(0, 0, 0, 0) for _ in range(len(hyp) + 1)] for _ in range(len(ref) + 1)]
    for index in range(1, len(ref) + 1):
        table[index][0] = (index, 0, index, 0)
    for index in range(1, len(hyp) + 1):
        table[0][index] = (index, 0, 0, index)

    for ref_index in range(1, len(ref) + 1):
        for hyp_index in range(1, len(hyp) + 1):
            if ref[ref_index - 1] == hyp[hyp_index - 1]:
                table[ref_index][hyp_index] = table[ref_index - 1][hyp_index - 1]
                continue
            diagonal = table[ref_index - 1][hyp_index - 1]
            deletion = table[ref_index - 1][hyp_index]
            insertion = table[ref_index][hyp_index - 1]
            candidates = [
                (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3]),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
            ]
            table[ref_index][hyp_index] = min(candidates)
    _, substitutions, deletions, insertions = table[-1][-1]
    return substitutions, deletions, insertions


def parse_timeline(markdown: str) -> list[dict]:
    pauses = []
    for line in (markdown or "").splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 7 or "→" not in parts[1]:
            continue
        try:
            pauses.append(
                {
                    "position": parts[1],
                    "start_s": float(parts[2].removesuffix("s")),
                    "duration_s": float(parts[3].removesuffix("s")),
                    "interpretation": parts[4],
                    "confidence": float(parts[5].removesuffix("%")) / 100.0,
                }
            )
        except ValueError:
            continue
    return pauses


def parse_rerank_status(markdown: str, hypothesis: str) -> dict:
    """Recover public reranking diagnostics from the Gradio Markdown output."""
    status = markdown or ""
    selected = re.search(r"selected candidate\s+(\d+)", status, flags=re.IGNORECASE)
    confidence = re.search(r"confidence\s+(\d+(?:\.\d+)?)%", status, flags=re.IGNORECASE)
    top1_text = hypothesis
    for line in status.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) >= 7 and parts[1] == "1":
            top1_text = parts[5]
            break
    error = ""
    if "Language reranking:" in status and "⚠️" in status:
        error = re.sub(r"^.*?Language reranking:\*\*\s*", "", status).strip()
    return {
        "original_text": top1_text,
        "selected_rank": int(selected.group(1)) if selected else 1,
        "changed": bool(selected and int(selected.group(1)) != 1),
        "confidence": float(confidence.group(1)) / 100.0 if confidence else "",
        "reason": status.splitlines()[-1].strip() if status.strip() else "",
        "error": error,
        "status": status,
    }


def summarize(rows: list[dict]) -> dict:
    valid = [row for row in rows if not row["error"]]
    total_words = sum(row["reference_words"] for row in valid)
    total_word_errors = sum(row["word_errors"] for row in valid)
    total_chars = sum(row["reference_chars"] for row in valid)
    total_char_errors = sum(row["char_errors"] for row in valid)
    expected_pauses = [row for row in valid if row["expected_pause_s"] != ""]
    detected_pauses = [row for row in expected_pauses if row["matched_pause_s"] != ""]
    classified_pauses = [row for row in detected_pauses if row["pause_mode_correct"] != ""]
    no_long_pause_expected = [row for row in valid if row["expected_pause_s"] == ""]
    unexpected_long = [row for row in no_long_pause_expected if row["long_pause_count"] > 0]

    groups = {}
    for group in sorted({row["group"] for row in valid}):
        group_rows = [row for row in valid if row["group"] == group]
        group_words = sum(row["reference_words"] for row in group_rows)
        groups[group] = {
            "clips": len(group_rows),
            "wer": (
                sum(row["word_errors"] for row in group_rows) / group_words
                if group_words
                else None
            ),
            "exact_sentence_rate": (
                sum(row["exact"] for row in group_rows) / len(group_rows)
                if group_rows
                else None
            ),
        }

    return {
        "clips_total": len(rows),
        "clips_successful": len(valid),
        "clips_failed": len(rows) - len(valid),
        "wer": total_word_errors / total_words if total_words else None,
        "word_accuracy_approx": 1.0 - total_word_errors / total_words if total_words else None,
        "cer": total_char_errors / total_chars if total_chars else None,
        "exact_sentence_rate": (
            sum(row["exact"] for row in valid) / len(valid) if valid else None
        ),
        "pause_detection_rate": (
            len(detected_pauses) / len(expected_pauses) if expected_pauses else None
        ),
        "pause_duration_mae_ms": (
            sum(row["pause_duration_error_ms"] for row in detected_pauses)
            / len(detected_pauses)
            if detected_pauses
            else None
        ),
        "pause_mode_accuracy": (
            sum(row["pause_mode_correct"] for row in classified_pauses)
            / len(classified_pauses)
            if classified_pauses
            else None
        ),
        "unexpected_long_pause_rate": (
            len(unexpected_long) / len(no_long_pause_expected)
            if no_long_pause_expected
            else None
        ),
        "mean_inference_s": (
            sum(row["inference_s"] for row in valid) / len(valid) if valid else None
        ),
        "groups": groups,
    }


def write_reports(rows: list[dict], output_csv: Path, output_json: Path) -> None:
    if rows:
        with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    output_json.write_text(
        json.dumps(summarize(rows), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable visually constrained language reranking.",
    )
    parser.add_argument(
        "--rerank-api-base",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--rerank-model", default="qwen3.7-flash")
    parser.add_argument(
        "--rerank-api-key-env",
        default="DASHSCOPE_API_KEY",
        help="Environment variable containing the session-only API key.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Load one local model instead of calling a running Gradio server.",
    )
    args = parser.parse_args()

    rerank_api_key = ""
    if args.rerank:
        rerank_api_key = os.environ.get(args.rerank_api_key_env, "").strip()
        if not rerank_api_key:
            raise RuntimeError(
                f"Language reranking is enabled, but {args.rerank_api_key_env} is empty."
            )

    manifest_path = args.manifest.resolve()
    output_csv = (args.output or manifest_path.with_name("accuracy_results.csv")).resolve()
    output_json = output_csv.with_suffix(".summary.json")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        manifest = list(csv.DictReader(file))

    missing = [row["filename"] for row in manifest if not (manifest_path.parent / row["filename"]).is_file()]
    if missing:
        raise FileNotFoundError("Missing test videos: " + ", ".join(missing))

    service = None
    client = None
    if args.local:
        from gradio_app import VSRService

        service = VSRService()
        if service.model is None:
            raise RuntimeError(service.startup_error)
    else:
        from gradio_client import Client

        client = Client(args.url)
    rows = []
    for index, sample in enumerate(manifest, start=1):
        video_path = manifest_path.parent / sample["filename"]
        started = time.perf_counter()
        state = {}
        rerank_status = ""
        print(f"[{index:02d}/{len(manifest):02d}] {video_path.name}", flush=True)
        try:
            if service is not None:
                response = service.recognize(
                    str(video_path),
                    "mediapipe",
                    args.beam,
                    0.1,
                    False,
                    "auto",
                    "natural",
                    True,
                    "auto",
                    "",
                    "",
                    args.rerank,
                    args.rerank_api_base,
                    args.rerank_model,
                    rerank_api_key,
                )
                hypothesis = response[1] or ""
                timeline = response[3] or ""
                state = response[4] or {}
                guidance = response[6] or ""
                rerank_status = response[8] or ""
                pauses = []
                for pause in state.get("pauses", []):
                    duration_s = float(pause["duration_s"])
                    if duration_s <= 0.30:
                        render_mode = "short"
                    elif (
                        float(pause["blank_confidence"]) >= 0.82
                        and float(pause["inactivity"]) >= 0.80
                    ):
                        render_mode = "breath"
                    else:
                        render_mode = "tail"
                    pauses.append(
                        {
                            "position": (
                                f"word {int(pause['after_word']) + 1} → "
                                f"word {int(pause['before_word']) + 1}"
                            ),
                            "start_s": float(pause["start_s"]),
                            "duration_s": duration_s,
                            "interpretation": pause["kind"],
                            "confidence": float(pause["confidence"]),
                            "blank_confidence": float(pause["blank_confidence"]),
                            "inactivity": float(pause["inactivity"]),
                            "render_mode": render_mode,
                        }
                    )
            else:
                from gradio_client import handle_file

                response = client.predict(
                    handle_file(str(video_path)),
                    "mediapipe",
                    args.beam,
                    0.1,
                    False,
                    "auto",
                    "natural",
                    True,
                    "auto",
                    "",
                    "",
                    args.rerank,
                    args.rerank_api_base,
                    args.rerank_model,
                    rerank_api_key,
                    api_name="/recognize_upload",
                )
                hypothesis = response[1] or ""
                timeline = response[3] or ""
                guidance = response[5] or ""
                rerank_status = response[7] or ""
                pauses = parse_timeline(timeline)
            error = guidance if not hypothesis else ""
        except Exception as exception:
            hypothesis = ""
            pauses = []
            guidance = ""
            error = f"{type(exception).__name__}: {exception}"

        reranking = state.get("reranking", {}) if isinstance(state, dict) else {}
        if args.rerank and not reranking:
            reranking = parse_rerank_status(rerank_status, hypothesis)

        ref_words = normalize_words(sample["reference"])
        hyp_words = normalize_words(hypothesis)
        substitutions, deletions, insertions = edit_counts(ref_words, hyp_words)
        word_errors = substitutions + deletions + insertions
        ref_chars = list("".join(ref_words))
        hyp_chars = list("".join(hyp_words))
        char_subs, char_deletes, char_inserts = edit_counts(ref_chars, hyp_chars)
        char_errors = char_subs + char_deletes + char_inserts
        expected_pause = (
            float(sample["expected_pause_s"]) if sample["expected_pause_s"] else None
        )
        long_pauses = [pause for pause in pauses if pause["duration_s"] > 0.30]
        matched_pause = (
            min(long_pauses, key=lambda pause: abs(pause["duration_s"] - expected_pause))
            if expected_pause is not None and long_pauses
            else None
        )
        row = {
            "id": int(sample["id"]),
            "filename": sample["filename"],
            "group": sample["group"],
            "reference": " ".join(ref_words),
            "visual_top1": " ".join(
                normalize_words(reranking.get("original_text", hypothesis))
            ),
            "hypothesis": " ".join(hyp_words),
            "rerank_changed": int(bool(reranking.get("changed", False))),
            "rerank_selected_rank": reranking.get("selected_rank", ""),
            "rerank_confidence": reranking.get("confidence", ""),
            "rerank_reason": reranking.get("reason", ""),
            "rerank_error": reranking.get("error", ""),
            "rerank_status": rerank_status,
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
            "word_errors": word_errors,
            "reference_words": len(ref_words),
            "wer": word_errors / len(ref_words) if ref_words else "",
            "reference_chars": len(ref_chars),
            "char_errors": char_errors,
            "cer": char_errors / len(ref_chars) if ref_chars else "",
            "exact": int(ref_words == hyp_words),
            "pause_count": len(pauses),
            "long_pause_count": len(long_pauses),
            "detected_pauses": json.dumps(pauses, ensure_ascii=False),
            "expected_pause_s": expected_pause if expected_pause is not None else "",
            "expected_pause_mode": sample["expected_pause_mode"],
            "matched_pause_s": matched_pause["duration_s"] if matched_pause else "",
            "matched_pause_mode": matched_pause.get("render_mode", "") if matched_pause else "",
            "pause_mode_correct": (
                int(matched_pause.get("render_mode") == sample["expected_pause_mode"])
                if matched_pause
                and matched_pause.get("render_mode")
                and sample["expected_pause_mode"]
                else ""
            ),
            "pause_duration_error_ms": (
                round(abs(matched_pause["duration_s"] - expected_pause) * 1000)
                if matched_pause and expected_pause is not None
                else ""
            ),
            "inference_s": round(time.perf_counter() - started, 3),
            "guidance": guidance,
            "error": error,
        }
        rows.append(row)
        write_reports(rows, output_csv, output_json)
        print(
            f"  REF={row['reference']} | HYP={row['hypothesis'] or '<EMPTY>'} | "
            f"WER={row['wer']:.1%} | long_pauses={len(long_pauses)}",
            flush=True,
        )

    print(json.dumps(summarize(rows), ensure_ascii=True, indent=2), flush=True)
    print(f"CSV: {output_csv}", flush=True)
    print(f"SUMMARY: {output_json}", flush=True)


if __name__ == "__main__":
    main()
