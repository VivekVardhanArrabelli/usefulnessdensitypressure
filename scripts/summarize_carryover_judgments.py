"""Summarize completed blinded carryover judgments."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JUDGMENTS = REPO_ROOT / "outputs" / "carryover_judge_sheet.csv"
DEFAULT_MAP = REPO_ROOT / "outputs" / "carryover_condition_map.csv"
DEFAULT_JOINED_OUT = REPO_ROOT / "outputs" / "carryover_judgments_joined.csv"
DEFAULT_SUMMARY_OUT = REPO_ROOT / "outputs" / "carryover_judgment_summary.md"

POSITIVE_AXES = [
    "overall_quality",
    "novelty",
    "useful_risk_taking",
    "assumption_quality",
    "uncertainty_quality",
    "actionability",
    "factual_grounding",
]
PENALTY_AXES = ["verbosity_penalty"]
REQUIRED_FIELDS = POSITIVE_AXES + PENALTY_AXES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize completed carryover judgments.")
    parser.add_argument("--judgments", default=str(DEFAULT_JUDGMENTS))
    parser.add_argument("--condition-map", default=str(DEFAULT_MAP))
    parser.add_argument("--joined-out", default=str(DEFAULT_JOINED_OUT))
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_OUT))
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
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(f"{row_hash}: {field} must be an integer, got {value!r}") from exc


def validate_judgments(judgment_rows: list[dict[str, str]], map_rows: list[dict[str, str]]) -> None:
    judgment_by_hash = row_hash_map(judgment_rows, "judgments")
    map_by_hash = row_hash_map(map_rows, "condition_map")
    if set(judgment_by_hash) != set(map_by_hash):
        missing_from_map = sorted(set(judgment_by_hash) - set(map_by_hash))
        missing_from_judgments = sorted(set(map_by_hash) - set(judgment_by_hash))
        raise ValueError(
            "row_hash sets differ. "
            f"Missing from condition map: {missing_from_map}; "
            f"missing from judgments: {missing_from_judgments}"
        )

    errors: list[str] = []
    for row in judgment_rows:
        row_hash = row["row_hash"]
        for field in REQUIRED_FIELDS:
            if not row.get(field, "").strip():
                errors.append(f"{row_hash}: missing {field}")
                continue
            try:
                value = parse_int(row[field], field=field, row_hash=row_hash)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if field in POSITIVE_AXES and not 1 <= value <= 5:
                errors.append(f"{row_hash}: {field} must be 1-5, got {value}")
            if field in PENALTY_AXES and not 0 <= value <= 3:
                errors.append(f"{row_hash}: {field} must be 0-3, got {value}")

    if errors:
        preview = "\n".join(errors[:30])
        remaining = len(errors) - 30
        suffix = f"\n... {remaining} more error(s)" if remaining > 0 else ""
        raise ValueError(f"Judgment validation failed:\n{preview}{suffix}")


def joined_rows(judgment_rows: list[dict[str, str]], map_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    map_by_hash = row_hash_map(map_rows, "condition_map")
    rows: list[dict[str, str]] = []
    for row in judgment_rows:
        map_row = map_by_hash[row["row_hash"]]
        joined = {**row, **{key: value for key, value in map_row.items() if key != "row_hash"}}
        joined["net_score"] = str(net_score(joined))
        rows.append(joined)
    return rows


def net_score(row: dict[str, str]) -> int:
    positives = sum(int(row[field]) for field in POSITIVE_AXES)
    penalties = sum(int(row[field]) for field in PENALTY_AXES)
    return positives - penalties


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


def fmt(value: float) -> str:
    return f"{value:.3f}"


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def condition_summary(rows: list[dict[str, str]]) -> list[str]:
    by_condition: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)

    lines = [
        "## Condition Means",
        "",
        "| condition | n | net_score | overall | novelty | risk | assumptions | uncertainty | actionability | grounding | verbosity_penalty |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, condition_rows in sorted(by_condition.items()):
        lines.append(
            f"| {condition} | {len(condition_rows)} | "
            f"{fmt(mean([float(row['net_score']) for row in condition_rows]))} | "
            f"{fmt(mean([float(row['overall_quality']) for row in condition_rows]))} | "
            f"{fmt(mean([float(row['novelty']) for row in condition_rows]))} | "
            f"{fmt(mean([float(row['useful_risk_taking']) for row in condition_rows]))} | "
            f"{fmt(mean([float(row['assumption_quality']) for row in condition_rows]))} | "
            f"{fmt(mean([float(row['uncertainty_quality']) for row in condition_rows]))} | "
            f"{fmt(mean([float(row['actionability']) for row in condition_rows]))} | "
            f"{fmt(mean([float(row['factual_grounding']) for row in condition_rows]))} | "
            f"{fmt(mean([float(row['verbosity_penalty']) for row in condition_rows]))} |"
        )
    return lines


def domain_summary(rows: list[dict[str, str]]) -> list[str]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["domain"], row["condition"])].append(row)

    lines = [
        "## Domain Means",
        "",
        "| domain | condition | n | net_score | overall |",
        "|---|---|---:|---:|---:|",
    ]
    for (domain, condition), group_rows in sorted(grouped.items()):
        lines.append(
            f"| {domain} | {condition} | {len(group_rows)} | "
            f"{fmt(mean([float(row['net_score']) for row in group_rows]))} | "
            f"{fmt(mean([float(row['overall_quality']) for row in group_rows]))} |"
        )
    return lines


def winner_summary(rows: list[dict[str, str]]) -> list[str]:
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)

    winner_counts: Counter[str] = Counter()
    lines = [
        "## Task Winners",
        "",
        "| task_id | domain | winner_by_net_score | winning_net_score | tied_conditions |",
        "|---|---|---|---:|---|",
    ]
    for task_id, task_rows in sorted(by_task.items()):
        max_score = max(int(row["net_score"]) for row in task_rows)
        winners = sorted(row["condition"] for row in task_rows if int(row["net_score"]) == max_score)
        winner_label = winners[0] if len(winners) == 1 else "tie"
        winner_counts[winner_label] += 1
        lines.append(
            f"| {task_id} | {task_rows[0]['domain']} | {winner_label} | {max_score} | "
            f"{', '.join(winners)} |"
        )
    lines.extend(["", "### Winner Counts", "", "| winner | count |", "|---|---:|"])
    for winner, count in sorted(winner_counts.items()):
        lines.append(f"| {winner} | {count} |")
    return lines


def build_summary(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Carryover Judgment Summary",
        "",
        "`net_score` is the sum of seven 1-5 judgment axes minus `verbosity_penalty`.",
        "",
    ]
    lines.extend(condition_summary(rows))
    lines.append("")
    lines.extend(domain_summary(rows))
    lines.append("")
    lines.extend(winner_summary(rows))
    lines.append("")
    return "\n".join(lines)


def write_text(path: str | Path, content: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    judgment_rows = load_csv(args.judgments)
    map_rows = load_csv(args.condition_map)
    try:
        validate_judgments(judgment_rows, map_rows)
        rows = joined_rows(judgment_rows, map_rows)
        write_csv(args.joined_out, rows)
        write_text(args.summary_out, build_summary(rows))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Wrote joined judgments to {args.joined_out}")
    print(f"Wrote summary to {args.summary_out}")


if __name__ == "__main__":
    main()
