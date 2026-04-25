"""Summarize completed productive-moves scoring sheets."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = REPO_ROOT / "outputs"

MOVE_PREFIXES = ["constraints", "alternatives", "approach", "uncertainty"]
REQUIRED_SCORE_FIELDS = [
    "correctness",
    "constraints_present",
    "constraints_appropriate",
    "alternatives_present",
    "alternatives_appropriate",
    "approach_present",
    "approach_appropriate",
    "uncertainty_present",
    "uncertainty_appropriate",
    "move_score",
    "total_score",
    "overframing_flag",
]

COMPARISONS = [
    ("structural_vs_vanilla42", "wandered_structural42", "vanilla42"),
    ("abstract_vs_vanilla42", "wandered_abstract42", "vanilla42"),
    ("vanilla43_vs_vanilla42", "vanilla43", "vanilla42"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a completed productive-moves score sheet.")
    parser.add_argument(
        "--scores",
        default=str(DEFAULT_OUTPUTS / "7b_productive_moves_scores.csv"),
    )
    parser.add_argument(
        "--condition-map",
        default=str(DEFAULT_OUTPUTS / "7b_productive_moves_condition_map.csv"),
    )
    parser.add_argument(
        "--joined-out",
        default=str(DEFAULT_OUTPUTS / "7b_productive_moves_joined.csv"),
    )
    parser.add_argument(
        "--summary-out",
        default=str(DEFAULT_OUTPUTS / "7b_productive_moves_summary.md"),
    )
    return parser.parse_args()


def load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def row_hash_map(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    mapped: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows, start=2):
        row_hash = row.get("row_hash", "").strip()
        if not row_hash:
            raise ValueError(f"{label}:{index} missing row_hash")
        if row_hash in mapped:
            raise ValueError(f"{label}:{index} duplicate row_hash {row_hash}")
        mapped[row_hash] = row
    return mapped


def parse_int(value: str, *, field: str, row_hash: str) -> int:
    stripped = value.strip()
    try:
        return int(stripped)
    except ValueError as exc:
        raise ValueError(f"{row_hash}: {field} must be an integer, got {value!r}") from exc


def parse_binary(value: str, *, field: str, row_hash: str) -> int:
    parsed = parse_int(value, field=field, row_hash=row_hash)
    if parsed not in {0, 1}:
        raise ValueError(f"{row_hash}: {field} must be 0 or 1, got {value!r}")
    return parsed


def move_contribution(row: dict[str, str], prefix: str) -> int:
    row_hash = row["row_hash"]
    present = parse_binary(row[f"{prefix}_present"], field=f"{prefix}_present", row_hash=row_hash)
    appropriate = parse_binary(
        row[f"{prefix}_appropriate"],
        field=f"{prefix}_appropriate",
        row_hash=row_hash,
    )
    if present == 0:
        return 0
    return 1 if appropriate == 1 else -1


def validate_scores(score_rows: list[dict[str, str]], map_rows: list[dict[str, str]]) -> None:
    scores_by_hash = row_hash_map(score_rows, "scores")
    map_by_hash = row_hash_map(map_rows, "condition_map")
    if set(scores_by_hash) != set(map_by_hash):
        missing_from_map = sorted(set(scores_by_hash) - set(map_by_hash))
        missing_from_scores = sorted(set(map_by_hash) - set(scores_by_hash))
        raise ValueError(
            "row_hash sets differ. "
            f"Missing from condition map: {missing_from_map}; missing from scores: {missing_from_scores}"
        )

    errors: list[str] = []
    for row in score_rows:
        row_hash = row["row_hash"]
        missing_fields = []
        for field in REQUIRED_SCORE_FIELDS:
            if not row.get(field, "").strip():
                missing_fields.append(field)
        if missing_fields:
            errors.extend(f"{row_hash}: missing {field}" for field in missing_fields)
            continue
        try:
            correctness = parse_binary(row["correctness"], field="correctness", row_hash=row_hash)
            parse_binary(row["overframing_flag"], field="overframing_flag", row_hash=row_hash)
            expected_move_score = sum(move_contribution(row, prefix) for prefix in MOVE_PREFIXES)
            move_score = parse_int(row["move_score"], field="move_score", row_hash=row_hash)
            if move_score != expected_move_score:
                errors.append(
                    f"{row_hash}: move_score={move_score}, expected {expected_move_score}"
                )
            total_score = parse_int(row["total_score"], field="total_score", row_hash=row_hash)
            expected_total_score = (3 * correctness) + move_score
            if total_score != expected_total_score:
                errors.append(
                    f"{row_hash}: total_score={total_score}, expected {expected_total_score}"
                )
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        preview = "\n".join(errors[:30])
        remaining = len(errors) - 30
        suffix = f"\n... {remaining} more error(s)" if remaining > 0 else ""
        raise ValueError(f"Score validation failed:\n{preview}{suffix}")


def joined_rows(score_rows: list[dict[str, str]], map_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    map_by_hash = row_hash_map(map_rows, "condition_map")
    joined: list[dict[str, str]] = []
    for score_row in score_rows:
        map_row = map_by_hash[score_row["row_hash"]]
        joined.append({**score_row, **{k: v for k, v in map_row.items() if k != "row_hash"}})
    return joined


def write_csv(path: str | Path, rows: list[dict[str, str]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No joined rows to write")
    fieldnames = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else math.nan


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    x_diffs = [x - x_mean for x in xs]
    y_diffs = [y - y_mean for y in ys]
    x_norm = math.sqrt(sum(value * value for value in x_diffs))
    y_norm = math.sqrt(sum(value * value for value in y_diffs))
    if x_norm == 0 or y_norm == 0:
        return None
    return sum(x * y for x, y in zip(x_diffs, y_diffs)) / (x_norm * y_norm)


def fmt(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.3f}"


def outcome(delta: int) -> str:
    if delta >= 2:
        return "win"
    if delta <= -2:
        return "loss"
    return "tie"


def rows_by_prompt_condition(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[row["prompt_id"]][row["condition"]] = row
    return grouped


def comparison_records(rows: list[dict[str, str]], candidate: str, baseline: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for prompt_id, by_condition in sorted(rows_by_prompt_condition(rows).items()):
        if candidate not in by_condition or baseline not in by_condition:
            raise ValueError(f"{prompt_id}: missing {candidate} or {baseline}")
        candidate_row = by_condition[candidate]
        baseline_row = by_condition[baseline]
        score_delta = int(candidate_row["total_score"]) - int(baseline_row["total_score"])
        token_delta = int(candidate_row["output_token_count"]) - int(baseline_row["output_token_count"])
        records.append(
            {
                "prompt_id": prompt_id,
                "category": candidate_row["category"],
                "score_delta": score_delta,
                "token_delta": token_delta,
                "outcome": outcome(score_delta),
            }
        )
    return records


def comparison_section(name: str, records: list[dict[str, Any]]) -> list[str]:
    score_deltas = [float(record["score_delta"]) for record in records]
    token_deltas = [float(record["token_delta"]) for record in records]
    outcomes = Counter(record["outcome"] for record in records)
    corr = pearson(token_deltas, score_deltas)
    lines = [
        f"### {name}",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| n | {len(records)} |",
        f"| mean_score_delta | {fmt(mean(score_deltas))} |",
        f"| stdev_score_delta | {fmt(stdev(score_deltas))} |",
        f"| wins | {outcomes['win']} |",
        f"| losses | {outcomes['loss']} |",
        f"| ties | {outcomes['tie']} |",
        f"| mean_token_delta | {fmt(mean(token_deltas))} |",
        f"| stdev_token_delta | {fmt(stdev(token_deltas))} |",
        f"| token_delta_score_delta_corr | {fmt(corr)} |",
        "",
        "| category | n | wins | losses | ties | mean_score_delta | mean_token_delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)
    for category, category_records in sorted(by_category.items()):
        category_outcomes = Counter(record["outcome"] for record in category_records)
        category_score_deltas = [float(record["score_delta"]) for record in category_records]
        category_token_deltas = [float(record["token_delta"]) for record in category_records]
        lines.append(
            f"| {category} | {len(category_records)} | {category_outcomes['win']} | "
            f"{category_outcomes['loss']} | {category_outcomes['tie']} | "
            f"{fmt(mean(category_score_deltas))} | {fmt(mean(category_token_deltas))} |"
        )
    return lines


def overframing_section(rows: list[dict[str, str]]) -> list[str]:
    counts: Counter[tuple[str, str]] = Counter()
    totals: Counter[str] = Counter()
    for row in rows:
        if int(row["overframing_flag"]) == 1:
            counts[(row["condition"], row["category"])] += 1
            totals[row["condition"]] += 1

    lines = [
        "## Overframing Counts",
        "",
        "| condition | category | count |",
        "|---|---|---:|",
    ]
    if not counts:
        lines.append("| none | none | 0 |")
    else:
        for (condition, category), count in sorted(counts.items()):
            lines.append(f"| {condition} | {category} | {count} |")
    lines.extend(["", "| condition | total |", "|---|---:|"])
    for condition in sorted({row["condition"] for row in rows}):
        lines.append(f"| {condition} | {totals[condition]} |")
    return lines


def rubric_issue_section(rows: list[dict[str, str]]) -> list[str]:
    issue_rows = [row for row in rows if row.get("rubric_issue", "").strip()]
    lines = [
        "## Rubric Issues",
        "",
        "| row_hash | prompt_id | category | condition | rubric_issue | notes |",
        "|---|---|---|---|---|---|",
    ]
    if not issue_rows:
        lines.append("| none | none | none | none | none | none |")
    else:
        for row in sorted(issue_rows, key=lambda item: (item["prompt_id"], item["condition"])):
            lines.append(
                f"| {row['row_hash']} | {row['prompt_id']} | {row['category']} | "
                f"{row['condition']} | {row['rubric_issue']} | {row.get('notes', '')} |"
            )
    return lines


def build_summary(rows: list[dict[str, str]]) -> str:
    lines = [
        "# 7B Productive-Moves Scoring Summary",
        "",
        "Score deltas use `candidate_total_score - baseline_total_score`.",
        "Win/loss/tie uses the rubric threshold: win >= 2, loss <= -2, tie otherwise.",
        "",
        "## Comparisons",
        "",
    ]
    for name, candidate, baseline in COMPARISONS:
        lines.extend(comparison_section(name, comparison_records(rows, candidate, baseline)))
        lines.append("")
    lines.extend(overframing_section(rows))
    lines.append("")
    lines.extend(rubric_issue_section(rows))
    lines.append("")
    return "\n".join(lines)


def write_text(path: str | Path, content: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    score_rows = load_csv(args.scores)
    map_rows = load_csv(args.condition_map)
    try:
        validate_scores(score_rows, map_rows)
        rows = joined_rows(score_rows, map_rows)
        write_csv(args.joined_out, rows)
        write_text(args.summary_out, build_summary(rows))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Wrote joined scores to {args.joined_out}")
    print(f"Wrote summary to {args.summary_out}")


if __name__ == "__main__":
    main()
