"""Summarize completed pairwise carryover judgments."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JUDGMENTS = REPO_ROOT / "outputs" / "carryover_pairwise_judge_sheet.csv"
DEFAULT_MAP = REPO_ROOT / "outputs" / "carryover_pairwise_condition_map.csv"
DEFAULT_JOINED_OUT = REPO_ROOT / "outputs" / "carryover_pairwise_joined.csv"
DEFAULT_SUMMARY_OUT = REPO_ROOT / "outputs" / "carryover_pairwise_summary.md"

VALID_WINNERS = {"a", "b", "tie"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize completed pairwise carryover judgments.")
    parser.add_argument("--judgments", default=str(DEFAULT_JUDGMENTS))
    parser.add_argument("--condition-map", default=str(DEFAULT_MAP))
    parser.add_argument("--joined-out", default=str(DEFAULT_JOINED_OUT))
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_OUT))
    return parser.parse_args()


def load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def map_by_pair_id(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    mapped: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows, start=2):
        pair_id = row.get("pair_id", "").strip()
        if not pair_id:
            raise ValueError(f"{label}:{index} missing pair_id")
        if pair_id in mapped:
            raise ValueError(f"{label}:{index} duplicate pair_id {pair_id}")
        mapped[pair_id] = row
    return mapped


def validate(judgment_rows: list[dict[str, str]], map_rows: list[dict[str, str]]) -> None:
    judgments = map_by_pair_id(judgment_rows, "judgments")
    condition_map = map_by_pair_id(map_rows, "condition_map")
    if set(judgments) != set(condition_map):
        missing_from_map = sorted(set(judgments) - set(condition_map))
        missing_from_judgments = sorted(set(condition_map) - set(judgments))
        raise ValueError(
            "pair_id sets differ. "
            f"Missing from condition map: {missing_from_map}; "
            f"missing from judgments: {missing_from_judgments}"
        )

    errors: list[str] = []
    for row in judgment_rows:
        pair_id = row["pair_id"]
        winner = row.get("winner", "").strip().lower()
        if winner not in VALID_WINNERS:
            errors.append(f"{pair_id}: winner must be one of a, b, tie")
        confidence_raw = row.get("confidence", "").strip()
        if not confidence_raw:
            errors.append(f"{pair_id}: missing confidence")
            continue
        try:
            confidence = int(confidence_raw)
        except ValueError:
            errors.append(f"{pair_id}: confidence must be an integer 1-3")
            continue
        if not 1 <= confidence <= 3:
            errors.append(f"{pair_id}: confidence must be 1-3")

    if errors:
        preview = "\n".join(errors[:30])
        remaining = len(errors) - 30
        suffix = f"\n... {remaining} more error(s)" if remaining > 0 else ""
        raise ValueError(f"Pairwise validation failed:\n{preview}{suffix}")


def joined_rows(judgment_rows: list[dict[str, str]], map_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    condition_map = map_by_pair_id(map_rows, "condition_map")
    rows: list[dict[str, str]] = []
    for row in judgment_rows:
        map_row = condition_map[row["pair_id"]]
        joined = {**row, **{key: value for key, value in map_row.items() if key != "pair_id"}}
        winner = row["winner"].strip().lower()
        if winner == "a":
            joined["winning_condition"] = map_row["condition_a"]
            joined["losing_condition"] = map_row["condition_b"]
        elif winner == "b":
            joined["winning_condition"] = map_row["condition_b"]
            joined["losing_condition"] = map_row["condition_a"]
        else:
            joined["winning_condition"] = "tie"
            joined["losing_condition"] = "tie"
        rows.append(joined)
    return rows


def write_csv(path: str | Path, rows: list[dict[str, str]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No joined rows to write")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def matchup_key(condition_a: str, condition_b: str) -> tuple[str, str]:
    return tuple(sorted([condition_a, condition_b]))


def matchup_summary(rows: list[dict[str, str]]) -> list[str]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[matchup_key(row["condition_a"], row["condition_b"])].append(row)

    lines = [
        "## Pairwise Win Rates",
        "",
        "| matchup | n | condition_1_wins | condition_2_wins | ties | headline |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for (condition_1, condition_2), group_rows in sorted(grouped.items()):
        wins_1 = sum(1 for row in group_rows if row["winning_condition"] == condition_1)
        wins_2 = sum(1 for row in group_rows if row["winning_condition"] == condition_2)
        ties = sum(1 for row in group_rows if row["winning_condition"] == "tie")
        if wins_1 > wins_2:
            headline = f"{condition_1} beats {condition_2} on {wins_1}/{len(group_rows)} tasks"
        elif wins_2 > wins_1:
            headline = f"{condition_2} beats {condition_1} on {wins_2}/{len(group_rows)} tasks"
        else:
            headline = f"{condition_1} and {condition_2} tie by task wins"
        lines.append(
            f"| {condition_1} vs {condition_2} | {len(group_rows)} | {wins_1} | "
            f"{wins_2} | {ties} | {headline} |"
        )
    return lines


def domain_summary(rows: list[dict[str, str]]) -> list[str]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        condition_1, condition_2 = matchup_key(row["condition_a"], row["condition_b"])
        grouped[(row["domain"], condition_1, condition_2)].append(row)

    lines = [
        "## Domain Pairwise Wins",
        "",
        "| domain | matchup | n | condition_1_wins | condition_2_wins | ties |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (domain, condition_1, condition_2), group_rows in sorted(grouped.items()):
        wins_1 = sum(1 for row in group_rows if row["winning_condition"] == condition_1)
        wins_2 = sum(1 for row in group_rows if row["winning_condition"] == condition_2)
        ties = sum(1 for row in group_rows if row["winning_condition"] == "tie")
        lines.append(
            f"| {domain} | {condition_1} vs {condition_2} | {len(group_rows)} | "
            f"{wins_1} | {wins_2} | {ties} |"
        )
    return lines


def condition_totals(rows: list[dict[str, str]]) -> list[str]:
    wins: Counter[str] = Counter()
    losses: Counter[str] = Counter()
    ties: Counter[str] = Counter()
    conditions = set()
    for row in rows:
        conditions.update([row["condition_a"], row["condition_b"]])
        if row["winning_condition"] == "tie":
            ties[row["condition_a"]] += 1
            ties[row["condition_b"]] += 1
        else:
            wins[row["winning_condition"]] += 1
            losses[row["losing_condition"]] += 1

    lines = [
        "## Condition Totals Across Pairwise Matchups",
        "",
        "| condition | wins | losses | ties |",
        "|---|---:|---:|---:|",
    ]
    for condition in sorted(conditions):
        lines.append(f"| {condition} | {wins[condition]} | {losses[condition]} | {ties[condition]} |")
    return lines


def task_rows(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "## Task-Level Pairwise Decisions",
        "",
        "| task_id | domain | condition_a | condition_b | winner | confidence | notes |",
        "|---|---|---|---|---|---:|---|",
    ]
    for row in sorted(rows, key=lambda item: (item["task_id"], item["condition_a"], item["condition_b"])):
        lines.append(
            f"| {row['task_id']} | {row['domain']} | {row['condition_a']} | {row['condition_b']} | "
            f"{row['winning_condition']} | {row['confidence']} | {row.get('notes', '')} |"
        )
    return lines


def build_summary(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Carryover Pairwise Judgment Summary",
        "",
        "Judgments compare blinded final `synthesize` artifacts for the same task.",
        "",
    ]
    lines.extend(matchup_summary(rows))
    lines.append("")
    lines.extend(condition_totals(rows))
    lines.append("")
    lines.extend(domain_summary(rows))
    lines.append("")
    lines.extend(task_rows(rows))
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
        validate(judgment_rows, map_rows)
        rows = joined_rows(judgment_rows, map_rows)
        write_csv(args.joined_out, rows)
        write_text(args.summary_out, build_summary(rows))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Wrote joined pairwise judgments to {args.joined_out}")
    print(f"Wrote pairwise summary to {args.summary_out}")


if __name__ == "__main__":
    main()
