"""Measure how much N-best reranking could improve deployed video-only VSR.

This evaluator reports both the normal top-1 result and an oracle result that
selects the candidate with the lowest WER from each N-best list.  The oracle is
not a deployable recognizer; it is a diagnostic upper bound for deciding
whether language-model reranking is worth adding.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
from pathlib import Path

import torch

from demo import decode_nbest, load_video_audio, preprocess_video
from evaluate_product import edit_counts, normalize_words
from gradio_app import VSRService


def word_error_count(reference: str, hypothesis: str) -> int:
    return sum(edit_counts(normalize_words(reference), normalize_words(hypothesis)))


def load_samples(manifests: list[Path]) -> list[dict]:
    samples = []
    for take, manifest in enumerate(manifests, start=1):
        manifest = manifest.resolve()
        with manifest.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                video_path = manifest.parent / row["filename"]
                if not video_path.is_file():
                    raise FileNotFoundError(f"Missing test video: {video_path}")
                samples.append(
                    {
                        "take": take,
                        "id": int(row["id"]),
                        "group": row["group"],
                        "reference": " ".join(normalize_words(row["reference"])),
                        "video_path": video_path,
                    }
                )
    return samples


def summarize(rows: list[dict]) -> dict:
    valid = [row for row in rows if not row["error"]]

    def metrics(selected: list[dict]) -> dict:
        words = sum(row["reference_words"] for row in selected)
        return {
            "clips": len(selected),
            "wer": sum(row["word_errors"] for row in selected) / words if words else None,
            "exact_sentence_rate": (
                sum(row["exact"] for row in selected) / len(selected)
                if selected
                else None
            ),
        }

    top1_rows = [
        {
            "reference_words": row["reference_words"],
            "word_errors": row["top1_word_errors"],
            "exact": row["top1_exact"],
        }
        for row in valid
    ]
    oracle_rows = [
        {
            "reference_words": row["reference_words"],
            "word_errors": row["oracle_word_errors"],
            "exact": row["oracle_exact"],
        }
        for row in valid
    ]
    groups = {}
    for group in sorted({row["group"] for row in valid}):
        group_rows = [row for row in valid if row["group"] == group]
        group_top1 = [
            {
                "reference_words": row["reference_words"],
                "word_errors": row["top1_word_errors"],
                "exact": row["top1_exact"],
            }
            for row in group_rows
        ]
        group_oracle = [
            {
                "reference_words": row["reference_words"],
                "word_errors": row["oracle_word_errors"],
                "exact": row["oracle_exact"],
            }
            for row in group_rows
        ]
        groups[group] = {
            "top1": metrics(group_top1),
            "oracle": metrics(group_oracle),
        }

    top1 = metrics(top1_rows)
    oracle = metrics(oracle_rows)
    return {
        "clips_total": len(rows),
        "clips_successful": len(valid),
        "clips_failed": len(rows) - len(valid),
        "beam_size": rows[0]["beam_size"] if rows else None,
        "top1": top1,
        "oracle_topk": oracle,
        "exact_sentence_gain_points": (
            (oracle["exact_sentence_rate"] - top1["exact_sentence_rate"]) * 100
            if top1["exact_sentence_rate"] is not None
            else None
        ),
        "relative_word_error_reduction": (
            (top1["wer"] - oracle["wer"]) / top1["wer"]
            if top1["wer"]
            else None
        ),
        "correct_answer_available_below_rank_1": sum(
            row["oracle_exact"] and not row["top1_exact"] for row in valid
        ),
        "groups": groups,
    }


def write_reports(rows: list[dict], output_csv: Path) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summarize(rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path


@torch.inference_mode()
def evaluate_sample(
    service: VSRService,
    sample: dict,
    beam_size: int,
    ctc_weight: float,
    top_k: int,
    scratch_dir: Path,
) -> dict:
    video_frames, _ = load_video_audio(str(sample["video_path"]))
    detector = service._get_detector("mediapipe")
    video_tensor = preprocess_video(
        video_frames,
        detector,
        service._video_processor,
        mouth_crop_path=str(scratch_dir / "mouth_crop.mp4"),
    )
    features = service.model.encoder(xs_v=video_tensor.unsqueeze(0).to(service.device))
    config = service._request_config(beam_size, ctc_weight)
    beam_search = service._get_beam_search(beam_size, ctc_weight)
    hypotheses = decode_nbest(
        features,
        beam_search,
        "v",
        config,
        max_candidates=top_k,
    )

    candidates = []
    for hypothesis in hypotheses:
        text = " ".join(normalize_words(hypothesis.text))
        errors = word_error_count(sample["reference"], text)
        candidates.append(
            {
                "rank": hypothesis.rank,
                "text": text,
                "token_ids": list(hypothesis.token_ids),
                "token_count": hypothesis.token_count,
                "score": hypothesis.score,
                "component_scores": hypothesis.component_scores,
                "word_errors": errors,
            }
        )
    if not candidates:
        raise RuntimeError("Beam search returned no non-empty hypotheses")

    oracle_rank, oracle = min(
        enumerate(candidates, start=1),
        key=lambda item: (item[1]["word_errors"], item[0]),
    )
    reference_words = len(normalize_words(sample["reference"]))
    return {
        "candidates": candidates,
        "reference_words": reference_words,
        "top1": candidates[0],
        "oracle": oracle,
        "oracle_rank": oracle_rank,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ctc-weight", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.top_k < 1 or args.top_k > args.beam:
        raise ValueError("--top-k must be between 1 and --beam")

    samples = load_samples(args.manifests)
    service = VSRService()
    if service.model is None:
        raise RuntimeError(service.startup_error)
    beam_size = min(args.beam, service.safe_beam_limit)
    if beam_size < args.beam:
        print(
            f"GPU guard reduced Beam {args.beam} to {beam_size}; "
            f"Top-k is now {min(args.top_k, beam_size)}.",
            flush=True,
        )
    top_k = min(args.top_k, beam_size)

    rows = []
    output_csv = args.output.resolve()
    try:
        with tempfile.TemporaryDirectory(prefix="usr2_nbest_") as temp_dir:
            scratch_dir = Path(temp_dir)
            for index, sample in enumerate(samples, start=1):
                started = time.perf_counter()
                print(
                    f"[{index:02d}/{len(samples):02d}] take={sample['take']} "
                    f"id={sample['id']:02d} {sample['video_path'].name}",
                    flush=True,
                )
                try:
                    result = evaluate_sample(
                        service,
                        sample,
                        beam_size,
                        args.ctc_weight,
                        top_k,
                        scratch_dir,
                    )
                    top1 = result["top1"]
                    oracle = result["oracle"]
                    error = ""
                except Exception as exception:
                    result = {"candidates": [], "reference_words": len(normalize_words(sample["reference"]))}
                    top1 = {"text": "", "word_errors": result["reference_words"]}
                    oracle = top1
                    result["oracle_rank"] = ""
                    error = f"{type(exception).__name__}: {exception}"

                row = {
                    "take": sample["take"],
                    "id": sample["id"],
                    "group": sample["group"],
                    "filename": sample["video_path"].name,
                    "reference": sample["reference"],
                    "beam_size": beam_size,
                    "candidate_count": len(result["candidates"]),
                    "candidates_json": json.dumps(result["candidates"], ensure_ascii=False),
                    "top1": top1["text"],
                    "top1_word_errors": top1["word_errors"],
                    "top1_exact": int(top1["word_errors"] == 0),
                    "oracle": oracle["text"],
                    "oracle_rank": result["oracle_rank"],
                    "oracle_word_errors": oracle["word_errors"],
                    "oracle_exact": int(oracle["word_errors"] == 0),
                    "reference_words": result["reference_words"],
                    "inference_s": round(time.perf_counter() - started, 3),
                    "error": error,
                }
                rows.append(row)
                summary_path = write_reports(rows, output_csv)
                print(
                    f"  top1={row['top1'] or '<EMPTY>'} | "
                    f"oracle(rank {row['oracle_rank'] or '-'})={row['oracle'] or '<EMPTY>'}",
                    flush=True,
                )
    finally:
        service.close()

    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2), flush=True)
    print(f"CSV: {output_csv}", flush=True)
    print(f"SUMMARY: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
