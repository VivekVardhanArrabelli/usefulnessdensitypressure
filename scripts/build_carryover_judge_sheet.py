"""Build a blinded judge sheet from carryover eval runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = REPO_ROOT / "outputs" / "carryover_runs.jsonl"
DEFAULT_JUDGE_OUT = REPO_ROOT / "outputs" / "carryover_judge_sheet.csv"
DEFAULT_MAP_OUT = REPO_ROOT / "outputs" / "carryover_condition_map.csv"

JUDGE_COLUMNS = [
    "row_hash",
    "task_id",
    "domain",
    "title",
    "seed_task",
    "final_artifact",
    "final_output",
    "final_output_token_count",
    "overall_quality",
    "novelty",
    "useful_risk_taking",
    "assumption_quality",
    "uncertainty_quality",
    "actionability",
    "factual_grounding",
    "verbosity_penalty",
    "notes",
]

MAP_COLUMNS = [
    "row_hash",
    "task_id",
    "condition",
    "model_label",
    "decode_config_hash",
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
    "model_label",
    "decode_config_hash",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a blinded carryover judge sheet.")
    parser.add_argument("--runs", default=str(DEFAULT_RUNS))
    parser.add_argument("--judge-out", default=str(DEFAULT_JUDGE_OUT))
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


def make_row_hash(task_id: str, condition: str, final_output: str) -> str:
    payload = "\n".join([task_id, condition, final_output])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_rows(runs: list[dict[str, Any]], shuffle_seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    judge_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for row in runs:
        row_hash = make_row_hash(str(row["task_id"]), str(row["condition"]), str(row["final_output"]))
        if row_hash in seen_hashes:
            raise ValueError(f"Duplicate row_hash generated: {row_hash}")
        seen_hashes.add(row_hash)

        judge_rows.append(
            {
                "row_hash": row_hash,
                "task_id": row["task_id"],
                "domain": row["domain"],
                "title": row["title"],
                "seed_task": row["seed_task"],
                "final_artifact": row["final_artifact"],
                "final_output": row["final_output"],
                "final_output_token_count": row["final_output_token_count"],
                "overall_quality": "",
                "novelty": "",
                "useful_risk_taking": "",
                "assumption_quality": "",
                "uncertainty_quality": "",
                "actionability": "",
                "factual_grounding": "",
                "verbosity_penalty": "",
                "notes": "",
            }
        )
        map_rows.append(
            {
                "row_hash": row_hash,
                "task_id": row["task_id"],
                "condition": row["condition"],
                "model_label": row["model_label"],
                "decode_config_hash": row["decode_config_hash"],
            }
        )

    random.Random(shuffle_seed).shuffle(judge_rows)
    return judge_rows, map_rows


def write_csv(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    judge_rows, map_rows = build_rows(load_runs(args.runs), args.shuffle_seed)
    write_csv(args.judge_out, judge_rows, JUDGE_COLUMNS)
    write_csv(args.condition_map_out, map_rows, MAP_COLUMNS)
    print(f"Wrote {len(judge_rows)} blinded judge rows to {args.judge_out}")
    print(f"Wrote {len(map_rows)} condition-map rows to {args.condition_map_out}")


if __name__ == "__main__":
    main()
