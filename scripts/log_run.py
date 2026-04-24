"""Small provenance helper for week-1 DPO runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 hex digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_config(output_dir: str | Path, config: dict[str, Any]) -> Path:
    """Write run_config.json with file hashes and a UTC timestamp."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = dict(config)
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    dataset_path = payload.get("dataset_path")
    if dataset_path:
        payload["dataset_sha256"] = sha256_file(dataset_path)

    decode_config_path = payload.get("decode_config_path")
    if decode_config_path:
        payload["decode_config_sha256"] = sha256_file(decode_config_path)

    output_path = out_dir / "run_config.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def _load_config_arg(value: str) -> dict[str, Any]:
    candidate = Path(value)
    if candidate.exists():
        with candidate.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    else:
        data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("--config-json must resolve to a JSON object")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a run_config.json file.")
    parser.add_argument("--output-dir", required=True, help="Directory for run_config.json.")
    parser.add_argument(
        "--config-json",
        required=True,
        help="JSON object or path to a JSON file containing run metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config_arg(args.config_json)
    path = write_run_config(args.output_dir, config)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
