"""Build a human scoring CSV from two JSONL eval output files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


CSV_COLUMNS = [
    "id",
    "category",
    "prompt",
    "base_output",
    "tuned_output",
    "base_tokens",
    "tuned_tokens",
    "correctness_base",
    "correctness_tuned",
    "naturalness_base",
    "naturalness_tuned",
    "context_richness_base",
    "context_richness_tuned",
    "rambling_base",
    "rambling_tuned",
    "winner",
    "notes",
]


def load_jsonl_by_id(path: str | Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get("id")
            if not row_id:
                raise ValueError(f"{path}:{line_number} is missing required field 'id'")
            if row_id in rows:
                raise ValueError(f"{path}:{line_number} duplicates id {row_id!r}")
            rows[row_id] = row
    if not rows:
        raise ValueError(f"{path} did not contain any JSONL rows")
    return rows


def require_same_ids(base: dict[str, Any], tuned: dict[str, Any]) -> list[str]:
    base_ids = set(base)
    tuned_ids = set(tuned)
    if base_ids != tuned_ids:
        missing_from_tuned = sorted(base_ids - tuned_ids)
        missing_from_base = sorted(tuned_ids - base_ids)
        raise ValueError(
            "Output files must contain the same ids. "
            f"Missing from tuned: {missing_from_tuned}; missing from base: {missing_from_base}"
        )
    return sorted(base_ids)


def check_decode_hash(
    base: dict[str, dict[str, Any]],
    tuned: dict[str, dict[str, Any]],
    *,
    allow_mismatch: bool,
    strict_match: bool,
) -> None:
    base_hashes = {row.get("decode_config_hash") for row in base.values() if row.get("decode_config_hash")}
    tuned_hashes = {row.get("decode_config_hash") for row in tuned.values() if row.get("decode_config_hash")}
    if base_hashes and tuned_hashes and base_hashes != tuned_hashes:
        message = f"Decode config hash mismatch: base={base_hashes}, tuned={tuned_hashes}"
        if strict_match:
            raise ValueError(message)
        prefix = "Allowed" if allow_mismatch else "WARNING"
        print(f"{prefix}: {message}", file=sys.stderr)


def _required(row: dict[str, Any], field: str, row_id: str, label: str) -> Any:
    value = row.get(field)
    if value is None:
        raise ValueError(f"{label} row {row_id!r} is missing required field {field!r}")
    return value


def build_rows(
    base: dict[str, dict[str, Any]],
    tuned: dict[str, dict[str, Any]],
    *,
    allow_decode_mismatch: bool = False,
    strict_decode_match: bool = False,
) -> list[dict[str, Any]]:
    ids = require_same_ids(base, tuned)
    check_decode_hash(
        base,
        tuned,
        allow_mismatch=allow_decode_mismatch,
        strict_match=strict_decode_match,
    )

    rows: list[dict[str, Any]] = []
    for row_id in ids:
        base_row = base[row_id]
        tuned_row = tuned[row_id]
        base_prompt = _required(base_row, "prompt", row_id, "base")
        tuned_prompt = _required(tuned_row, "prompt", row_id, "tuned")
        if base_prompt != tuned_prompt:
            raise ValueError(f"Prompt mismatch for id {row_id!r}")

        rows.append(
            {
                "id": row_id,
                "category": base_row.get("category", tuned_row.get("category", "")),
                "prompt": base_prompt,
                "base_output": _required(base_row, "output", row_id, "base"),
                "tuned_output": _required(tuned_row, "output", row_id, "tuned"),
                "base_tokens": _required(base_row, "output_token_count", row_id, "base"),
                "tuned_tokens": _required(tuned_row, "output_token_count", row_id, "tuned"),
                "correctness_base": "",
                "correctness_tuned": "",
                "naturalness_base": "",
                "naturalness_tuned": "",
                "context_richness_base": "",
                "context_richness_tuned": "",
                "rambling_base": "",
                "rambling_tuned": "",
                "winner": "",
                "notes": "",
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a base-vs-tuned scoring CSV.")
    parser.add_argument("--base", required=True, help="Base or first-condition JSONL output.")
    parser.add_argument("--tuned", required=True, help="Tuned or second-condition JSONL output.")
    parser.add_argument("--out", required=True, help="CSV path to write.")
    parser.add_argument(
        "--allow-decode-mismatch",
        "--allow_decode_mismatch",
        action="store_true",
        help="Mark a decode config mismatch as intentional; comparison still proceeds.",
    )
    parser.add_argument(
        "--strict-decode-match",
        action="store_true",
        help="Promote decode config hash mismatch to an error.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_jsonl_by_id(args.base)
    tuned = load_jsonl_by_id(args.tuned)
    rows = build_rows(
        base,
        tuned,
        allow_decode_mismatch=args.allow_decode_mismatch,
        strict_decode_match=args.strict_decode_match,
    )
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} comparison rows to {args.out}")


if __name__ == "__main__":
    main()
