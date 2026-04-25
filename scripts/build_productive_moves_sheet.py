"""Build a blinded productive-moves scoring sheet from eval outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = REPO_ROOT / "outputs"

SOURCES = [
    ("vanilla42", "7b_vanilla_42.jsonl"),
    ("vanilla43", "7b_vanilla_43.jsonl"),
    ("wandered_abstract42", "7b_wandered_abstract_42.jsonl"),
    ("wandered_structural42", "7b_wandered_structural_42.jsonl"),
]

SCORE_COLUMNS = [
    "row_hash",
    "prompt_id",
    "category",
    "prompt_text",
    "output_text",
    "output_token_count",
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
    "rubric_issue",
    "notes",
]

MAP_COLUMNS = [
    "row_hash",
    "prompt_id",
    "condition",
    "source_file",
    "model_label",
    "decode_config_hash",
]

REQUIRED_JSONL_FIELDS = {
    "id",
    "category",
    "prompt",
    "model_label",
    "output",
    "output_token_count",
    "decode_config_hash",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a blinded productive-moves scoring sheet.")
    parser.add_argument("--outputs-dir", default=str(DEFAULT_OUTPUTS))
    parser.add_argument(
        "--scores-out",
        default=str(DEFAULT_OUTPUTS / "7b_productive_moves_scores.csv"),
    )
    parser.add_argument(
        "--condition-map-out",
        default=str(DEFAULT_OUTPUTS / "7b_productive_moves_condition_map.csv"),
    )
    parser.add_argument("--shuffle-seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED_JSONL_FIELDS - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contained no rows")
    return rows


def make_row_hash(prompt_id: str, condition: str, output_text: str) -> str:
    payload = "\n".join([prompt_id, condition, output_text])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def validate_prompt_sets(rows_by_condition: dict[str, list[dict[str, Any]]]) -> None:
    expected_ids: set[str] | None = None
    for condition, rows in rows_by_condition.items():
        ids = [str(row["id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{condition} contains duplicate prompt ids")
        current_ids = set(ids)
        if expected_ids is None:
            expected_ids = current_ids
            continue
        if current_ids != expected_ids:
            raise ValueError(f"{condition} prompt ids differ from the first condition")


def build_rows(outputs_dir: Path, shuffle_seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_condition = {
        condition: load_jsonl(outputs_dir / filename) for condition, filename in SOURCES
    }
    validate_prompt_sets(rows_by_condition)

    score_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for condition, filename in SOURCES:
        for row in rows_by_condition[condition]:
            row_hash = make_row_hash(str(row["id"]), condition, str(row["output"]))
            if row_hash in seen_hashes:
                raise ValueError(f"Duplicate row_hash generated: {row_hash}")
            seen_hashes.add(row_hash)

            score_rows.append(
                {
                    "row_hash": row_hash,
                    "prompt_id": row["id"],
                    "category": row["category"],
                    "prompt_text": row["prompt"],
                    "output_text": row["output"],
                    "output_token_count": row["output_token_count"],
                    "correctness": "",
                    "constraints_present": "",
                    "constraints_appropriate": "",
                    "alternatives_present": "",
                    "alternatives_appropriate": "",
                    "approach_present": "",
                    "approach_appropriate": "",
                    "uncertainty_present": "",
                    "uncertainty_appropriate": "",
                    "move_score": "",
                    "total_score": "",
                    "overframing_flag": "",
                    "rubric_issue": "",
                    "notes": "",
                }
            )
            map_rows.append(
                {
                    "row_hash": row_hash,
                    "prompt_id": row["id"],
                    "condition": condition,
                    "source_file": str(Path("outputs") / filename),
                    "model_label": row["model_label"],
                    "decode_config_hash": row["decode_config_hash"],
                }
            )

    random.Random(shuffle_seed).shuffle(score_rows)
    return score_rows, map_rows


def write_csv(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    score_rows, map_rows = build_rows(Path(args.outputs_dir), args.shuffle_seed)
    write_csv(args.scores_out, score_rows, SCORE_COLUMNS)
    write_csv(args.condition_map_out, map_rows, MAP_COLUMNS)
    print(f"Wrote {len(score_rows)} blinded scoring rows to {args.scores_out}")
    print(f"Wrote {len(map_rows)} condition-map rows to {args.condition_map_out}")


if __name__ == "__main__":
    main()
