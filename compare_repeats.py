"""Compare two product-evaluation CSV files and produce a stability report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("take1", type=Path)
    parser.add_argument("take2", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    take1 = load_rows(args.take1.resolve())
    take2 = load_rows(args.take2.resolve())
    by_id_1 = {int(row["id"]): row for row in take1}
    by_id_2 = {int(row["id"]): row for row in take2}
    if set(by_id_1) != set(by_id_2):
        raise ValueError("The two result files do not contain the same sample IDs")

    comparisons = []
    for sample_id in sorted(by_id_1):
        first = by_id_1[sample_id]
        second = by_id_2[sample_id]
        if first["reference"] != second["reference"]:
            raise ValueError(f"Reference mismatch for sample {sample_id}")
        exact1 = int(first["exact"])
        exact2 = int(second["exact"])
        if exact1 and exact2:
            stability = "stable_correct"
        elif exact1:
            stability = "take1_only"
        elif exact2:
            stability = "take2_only"
        else:
            stability = "both_wrong"
        comparisons.append(
            {
                "id": sample_id,
                "group": first["group"],
                "reference": first["reference"],
                "take1_filename": first["filename"],
                "take1_hypothesis": first["hypothesis"],
                "take1_wer": first["wer"],
                "take1_exact": exact1,
                "take2_filename": second["filename"],
                "take2_hypothesis": second["hypothesis"],
                "take2_wer": second["wer"],
                "take2_exact": exact2,
                "same_prediction": int(first["hypothesis"] == second["hypothesis"]),
                "stability": stability,
            }
        )

    all_rows = take1 + take2
    total_reference_words = sum(int(row["reference_words"]) for row in all_rows)
    total_word_errors = sum(int(row["word_errors"]) for row in all_rows)
    total_reference_chars = sum(int(row["reference_chars"]) for row in all_rows)
    total_char_errors = sum(int(row["char_errors"]) for row in all_rows)
    expected_pauses = [row for row in all_rows if row["expected_pause_s"]]
    detected_pauses = [row for row in expected_pauses if row["matched_pause_s"]]
    classified_pauses = [row for row in detected_pauses if row["pause_mode_correct"]]
    no_pause_expected = [row for row in all_rows if not row["expected_pause_s"]]

    group_totals = defaultdict(lambda: {"clips": 0, "words": 0, "errors": 0, "exact": 0})
    for row in all_rows:
        group = group_totals[row["group"]]
        group["clips"] += 1
        group["words"] += int(row["reference_words"])
        group["errors"] += int(row["word_errors"])
        group["exact"] += int(row["exact"])

    counts = defaultdict(int)
    for row in comparisons:
        counts[row["stability"]] += 1
    summary = {
        "clips_total": len(all_rows),
        "unique_sentences": len(comparisons),
        "take1_wer": ratio(
            sum(int(row["word_errors"]) for row in take1),
            sum(int(row["reference_words"]) for row in take1),
        ),
        "take2_wer": ratio(
            sum(int(row["word_errors"]) for row in take2),
            sum(int(row["reference_words"]) for row in take2),
        ),
        "combined_wer": ratio(total_word_errors, total_reference_words),
        "combined_word_accuracy_approx": 1.0 - ratio(total_word_errors, total_reference_words),
        "combined_cer": ratio(total_char_errors, total_reference_chars),
        "combined_exact_sentence_rate": ratio(
            sum(int(row["exact"]) for row in all_rows), len(all_rows)
        ),
        "same_prediction_rate": ratio(
            sum(row["same_prediction"] for row in comparisons), len(comparisons)
        ),
        "stable_correct_rate": ratio(counts["stable_correct"], len(comparisons)),
        "at_least_one_exact_rate": ratio(
            counts["stable_correct"] + counts["take1_only"] + counts["take2_only"],
            len(comparisons),
        ),
        "stability_counts": dict(counts),
        "pause_detection_rate": ratio(len(detected_pauses), len(expected_pauses)),
        "pause_duration_mae_ms": ratio(
            sum(float(row["pause_duration_error_ms"]) for row in detected_pauses),
            len(detected_pauses),
        ),
        "pause_mode_accuracy": ratio(
            sum(int(row["pause_mode_correct"]) for row in classified_pauses),
            len(classified_pauses),
        ),
        "unexpected_long_pause_rate": ratio(
            sum(int(row["long_pause_count"]) > 0 for row in no_pause_expected),
            len(no_pause_expected),
        ),
        "groups": {
            name: {
                "clips": values["clips"],
                "wer": ratio(values["errors"], values["words"]),
                "exact_sentence_rate": ratio(values["exact"], values["clips"]),
            }
            for name, values in sorted(group_totals.items())
        },
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "repeat_comparison.csv"
    summary_path = output_dir / "combined_accuracy_summary.json"
    report_path = output_dir / "combined_accuracy_report.md"
    with comparison_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparisons[0].keys()))
        writer.writeheader()
        writer.writerows(comparisons)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    group_lines = "\n".join(
        f"| {name} | {values['clips']} | {percent(values['wer'])} | "
        f"{percent(values['exact_sentence_rate'])} |"
        for name, values in summary["groups"].items()
    )
    report = f"""# 两遍录制合并评测

测试规模：30 句 × 2 遍，共 60 段视频。识别设置为 USR 2.0 Huge、纯视觉、MediaPipe、Beam 5。

## 总体结果

| 指标 | 结果 |
|---|---:|
| 第一遍 WER | {percent(summary['take1_wer'])} |
| 第二遍 WER | {percent(summary['take2_wer'])} |
| 合并 WER | {percent(summary['combined_wer'])} |
| 合并近似词正确率 | {percent(summary['combined_word_accuracy_approx'])} |
| 合并 CER | {percent(summary['combined_cer'])} |
| 合并整句完全正确率 | {percent(summary['combined_exact_sentence_rate'])} |
| 两遍输出完全一致率 | {percent(summary['same_prediction_rate'])} |
| 两遍都完全正确 | {counts['stable_correct']} / 30 |
| 仅第一遍正确 | {counts['take1_only']} / 30 |
| 仅第二遍正确 | {counts['take2_only']} / 30 |
| 两遍都错误 | {counts['both_wrong']} / 30 |
| 至少一遍完全正确 | {percent(summary['at_least_one_exact_rate'])} |

## 合并分组

| 分组 | 视频数 | WER | 整句完全正确率 |
|---|---:|---:|---:|
{group_lines}

## 停顿

| 指标 | 结果 |
|---|---:|
| 指定长停顿检出率 | {percent(summary['pause_detection_rate'])} |
| 已检出停顿时长 MAE | {summary['pause_duration_mae_ms']:.0f} ms |
| 呼吸/尾音判断准确率 | {percent(summary['pause_mode_accuracy'])} |
| 非指定长停顿误检率 | {percent(summary['unexpected_long_pause_rate'])} |

## 解释

合并 WER 按两遍全部参考词汇总，不是简单平均每条百分比。单条 WER 可以超过 100%，因为插入错误会让错误词数超过参考词数。两遍输出完全一致率衡量重复录制稳定性；“两遍都完全正确”是更严格、更可信的产品指标。
"""
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(comparison_path)
    print(summary_path)
    print(report_path)


if __name__ == "__main__":
    main()
