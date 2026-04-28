"""Build a blinded pairwise judge sheet from carryover eval runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = REPO_ROOT / "outputs" / "carryover_runs.jsonl"
DEFAULT_PAIRWISE_OUT = REPO_ROOT / "outputs" / "carryover_pairwise_judge_sheet.csv"
DEFAULT_MAP_OUT = REPO_ROOT / "outputs" / "carryover_pairwise_condition_map.csv"

PAIRWISE_COLUMNS = [
    "pair_id",
    "task_id",
    "domain",
    "title",
    "seed_task",
    "final_artifact",
    "artifact_a",
    "artifact_a_token_count",
    "artifact_b",
    "artifact_b_token_count",
    "winner",
    "confidence",
    "notes",
]

MAP_COLUMNS = [
    "pair_id",
    "task_id",
    "condition_a",
    "condition_b",
    "row_hash_a",
    "row_hash_b",
]

REQUIRED_RUN_FIELDS = {
    "task_id",
    "domain",
    "title",
    "seed_task",
    "final_artifact",
    "condition",
    "final_output",
    "final_output_token_count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a blinded pairwise carryover judge sheet.")
    parser.add_argument("--runs", default=str(DEFAULT_RUNS))
    parser.add_argument("--pairwise-out", default=str(DEFAULT_PAIRWISE_OUT))
    parser.add_argument("--condition-map-out", default=str(DEFAULT_MAP_OUT))
    parser.add_argument("--shuffle-seed", type=int, default=42)
    return parser.parse_args()


def load_runs(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED_RUN_FIELDS - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contained no run rows")
    return rows


def row_hash(task_id: str, condition: str, final_output: str) -> str:
    payload = "\n".join([task_id, condition, final_output])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def pair_id_for(task_id: str, condition_a: str, condition_b: str, output_a: str, output_b: str) -> str:
    payload = "\n".join([task_id, condition_a, condition_b, output_a, output_b])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def group_by_task_condition(runs: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in runs:
        task_id = str(row["task_id"])
        condition = str(row["condition"])
        if condition in grouped[task_id]:
            raise ValueError(f"Duplicate run for task={task_id} condition={condition}")
        grouped[task_id][condition] = row
    return grouped


def build_rows(runs: list[dict[str, Any]], shuffle_seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(shuffle_seed)
    pairwise_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    seen_pair_ids: set[str] = set()

    for task_id, by_condition in sorted(group_by_task_condition(runs).items()):
        conditions = sorted(by_condition)
        if len(conditions) < 2:
            raise ValueError(f"{task_id} has fewer than two conditions")
        for left_condition, right_condition in itertools.combinations(conditions, 2):
            left = by_condition[left_condition]
            right = by_condition[right_condition]
            if rng.random() < 0.5:
                condition_a, row_a = left_condition, left
                condition_b, row_b = right_condition, right
            else:
                condition_a, row_a = right_condition, right
                condition_b, row_b = left_condition, left
            pair_id = pair_id_for(
                task_id,
                condition_a,
                condition_b,
                str(row_a["final_output"]),
                str(row_b["final_output"]),
            )
            if pair_id in seen_pair_ids:
                raise ValueError(f"Duplicate pair_id generated: {pair_id}")
            seen_pair_ids.add(pair_id)

            pairwise_rows.append(
                {
                    "pair_id": pair_id,
                    "task_id": task_id,
                    "domain": row_a["domain"],
                    "title": row_a["title"],
                    "seed_task": row_a["seed_task"],
                    "final_artifact": row_a["final_artifact"],
                    "artifact_a": row_a["final_output"],
                    "artifact_a_token_count": row_a["final_output_token_count"],
                    "artifact_b": row_b["final_output"],
                    "artifact_b_token_count": row_b["final_output_token_count"],
                    "winner": "",
                    "confidence": "",
                    "notes": "",
                }
            )
            map_rows.append(
                {
                    "pair_id": pair_id,
                    "task_id": task_id,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "row_hash_a": row_hash(task_id, condition_a, str(row_a["final_output"])),
                    "row_hash_b": row_hash(task_id, condition_b, str(row_b["final_output"])),
                }
            )

    rng.shuffle(pairwise_rows)
    return pairwise_rows, map_rows


def write_csv(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    pairwise_rows, map_rows = build_rows(load_runs(args.runs), args.shuffle_seed)
    write_csv(args.pairwise_out, pairwise_rows, PAIRWISE_COLUMNS)
    write_csv(args.condition_map_out, map_rows, MAP_COLUMNS)
    print(f"Wrote {len(pairwise_rows)} blinded pairwise rows to {args.pairwise_out}")
    print(f"Wrote {len(map_rows)} pairwise condition-map rows to {args.condition_map_out}")


if __name__ == "__main__":
    main()
