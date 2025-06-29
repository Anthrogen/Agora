import argparse
import csv
import json
import os
from pathlib import Path
from typing import List


def find_members(directory: Path, csv_dir: Path) -> List[List[str]]:

    rows: List[List[str]] = []

    for item in directory.iterdir():
        if item.is_file() and item.suffix.lower() == ".json":
            try:
                with item.open("r", encoding="utf‑8") as fh:
                    data = json.load(fh)
            except Exception:
                # Skip unreadable or invalid JSON files silently
                continue

            seq_id = data.get("sequence_id")
            if seq_id is None:
                continue

            rel_path = os.path.relpath(item, csv_dir)
            rows.insert(0, ["", str(seq_id), rel_path, ""])

    return rows


def write_csv(csv_path: Path, rows: List[List[str]]) -> None:
    """Write *rows* to *csv_path* with the required header and final newline."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["sequence_id", "rep_id", "rep_json_path", "member_json_path"]

    # Use newline="" so csv module controls line endings (RFC‑4180 compliant)
    with csv_path.open("w", newline="", encoding="utf‑8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file")
    parser.add_argument("directory")
    args = parser.parse_args()

    csv_path = Path(args.csv_file).resolve()
    directory = Path(args.directory).resolve()

    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")

    rows = find_members(directory, csv_path.parent)
    write_csv(csv_path, rows)


if __name__ == "__main__":
    main()
