"""Tune conservative visual/language fusion on a fixed scored N-best cache."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from language_reranker import RecognitionCandidate, visually_eligible_candidates


@dataclass(frozen=True)
class Parameters:
    language_weight: float
    max_visual_drop_per_token: float
    max_ctc_drop_per_token: float
    min_confidence: float
    override_margin: float


def candidate_from_dict(raw: dict, fallback_rank: int) -> RecognitionCandidate:
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


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        raw_rows = list(csv.DictReader(file))
    rows = []
    for raw in raw_rows:
        if raw.get("error"):
            continue
        candidate_dicts = json.loads(raw["candidates_json"])
        candidates = [
            candidate_from_dict(value, index)
            for index, value in enumerate(candidate_dicts, start=1)
        ]
        word_errors = {
            candidate.rank: int(value["word_errors"])
            for candidate, value in zip(candidates, candidate_dicts)
        }
        scores = {
            int(rank): float(score)
            for rank, score in json.loads(raw["language_scores_json"]).items()
        }
        rows.append(
            {
                "take": int(raw["take"]),
                "id": int(raw["id"]),
                "group": raw["group"],
                "reference": raw["reference"],
                "reference_words": int(raw["reference_words"]),
                "candidates": candidates,
                "word_errors": word_errors,
                "language_scores": scores,
                "confidence": float(raw["confidence"]),
            }
        )
    return rows


def select(row: dict, parameters: Parameters) -> RecognitionCandidate:
    candidates = row["candidates"]
    top1 = candidates[0]
    eligible, drops = visually_eligible_candidates(
        candidates,
        max_visual_drop_per_token=parameters.max_visual_drop_per_token,
        max_ctc_drop_per_token=parameters.max_ctc_drop_per_token,
    )
    eligible = [item for item in eligible if item.rank in row["language_scores"]]
    if len(eligible) < 2 or row["confidence"] < parameters.min_confidence:
        return top1
    top_language = row["language_scores"][top1.rank]
    fused = {
        item.rank: -drops[item.rank]
        + parameters.language_weight
        * (row["language_scores"][item.rank] - top_language)
        for item in eligible
    }
    winner = max(eligible, key=lambda item: (fused[item.rank], -item.rank))
    if winner.rank != top1.rank and fused[winner.rank] <= parameters.override_margin:
        return top1
    return winner


def metrics(rows: list[dict], parameters: Parameters | None = None) -> dict:
    words = errors = exact = changes = improved = worsened = 0
    details = []
    for row in rows:
        top1 = row["candidates"][0]
        selected = top1 if parameters is None else select(row, parameters)
        before = row["word_errors"][top1.rank]
        after = row["word_errors"][selected.rank]
        words += row["reference_words"]
        errors += after
        exact += after == 0
        changes += selected.rank != top1.rank
        improved += after < before
        worsened += after > before
        details.append(
            {
                "take": row["take"],
                "id": row["id"],
                "reference": row["reference"],
                "top1": top1.text,
                "selected": selected.text,
                "selected_rank": selected.rank,
                "before_errors": before,
                "after_errors": after,
            }
        )
    return {
        "clips": len(rows),
        "words": words,
        "word_errors": errors,
        "wer": errors / words if words else None,
        "exact_sentences": exact,
        "exact_sentence_rate": exact / len(rows) if rows else None,
        "changes": changes,
        "improved": improved,
        "worsened": worsened,
        "details": details,
    }


def score_key(result: dict, parameters: Parameters) -> tuple:
    return (
        result["word_errors"],
        result["worsened"],
        -result["exact_sentences"],
        result["changes"],
        parameters.language_weight,
        parameters.max_visual_drop_per_token,
        parameters.max_ctc_drop_per_token,
        -parameters.min_confidence,
        parameters.override_margin,
    )


def parameter_grid() -> list[Parameters]:
    return [
        Parameters(*values)
        for values in itertools.product(
            (0.10, 0.20, 0.35, 0.50, 0.75, 1.00),
            (0.10, 0.20, 0.35, 0.50, 0.75),
            (0.20, 0.35, 0.50, 0.75, 1.00),
            (0.50, 0.60, 0.70, 0.80, 0.90),
            (0.00, 0.02, 0.05, 0.10),
        )
    ]


def best_parameters(rows: list[dict], grid: list[Parameters]) -> tuple[Parameters, dict]:
    best = None
    best_result = None
    for parameters in grid:
        result = metrics(rows, parameters)
        if best is None or score_key(result, parameters) < score_key(best_result, best):
            best, best_result = parameters, result
    return best, best_result


def compact(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "details"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")

    rows = load_rows(args.input.resolve())
    if not rows:
        raise RuntimeError("No successfully scored rows were found")
    grid = parameter_grid()
    baseline = metrics(rows)
    full_parameters, full_result = best_parameters(rows, grid)

    held_out_details = []
    fold_reports = []
    for fold in range(args.folds):
        train = [row for row in rows if (row["id"] - 1) % args.folds != fold]
        held_out = [row for row in rows if (row["id"] - 1) % args.folds == fold]
        parameters, train_result = best_parameters(train, grid)
        held_out_result = metrics(held_out, parameters)
        held_out_details.extend(held_out_result["details"])
        fold_reports.append(
            {
                "fold": fold + 1,
                "held_out_sentence_ids": sorted({row["id"] for row in held_out}),
                "parameters": asdict(parameters),
                "train": compact(train_result),
                "held_out": compact(held_out_result),
            }
        )

    words = sum(row["reference_words"] for row in rows)
    cv_errors = sum(item["after_errors"] for item in held_out_details)
    cv_exact = sum(item["after_errors"] == 0 for item in held_out_details)
    cv = {
        "clips": len(held_out_details),
        "words": words,
        "word_errors": cv_errors,
        "wer": cv_errors / words,
        "exact_sentences": cv_exact,
        "exact_sentence_rate": cv_exact / len(held_out_details),
        "changes": sum(item["selected_rank"] != 1 for item in held_out_details),
        "improved": sum(item["after_errors"] < item["before_errors"] for item in held_out_details),
        "worsened": sum(item["after_errors"] > item["before_errors"] for item in held_out_details),
    }
    report = {
        "source": str(args.input.resolve()),
        "rows": len(rows),
        "unique_sentence_ids": len({row["id"] for row in rows}),
        "grid_size": len(grid),
        "baseline": compact(baseline),
        "cross_validation": cv,
        "recommended_parameters_fit_on_all_development_data": asdict(full_parameters),
        "fit_on_all_development_data": compact(full_result),
        "folds": fold_reports,
        "recommended_details": full_result["details"],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"folds", "recommended_details"}}, ensure_ascii=False, indent=2))
    print(f"REPORT: {output}")


if __name__ == "__main__":
    main()
