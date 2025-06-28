import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Edit this variable if you prefer to hard-code the directory containing the
# JSON files. You can also leave it as an empty string and supply the directory
# on the command line instead.
# ---------------------------------------------------------------------------
JSON_DIR: str = "1k"  # e.g. "../sample_data/100"

HEADER: List[str] = [
    "sequence_id",
    "rep_id",
    "rep_json_path",
    "member_json_path",
]


def extract_sequence_id(data: Dict[str, Any]) -> Optional[str]:
    """Return the `sequence_id` field from a JSON object if it exists.

    The field name is expected to be exactly "sequence_id". If it is missing
    the file will be skipped.
    """

    return data.get("sequence_id")


def build_index(directory: Path) -> int:
    """Generate `index.csv` inside *directory*.

    Returns the number of rows written (excluding the header).
    """

    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")

    rows: List[Dict[str, str]] = []
    for json_file in sorted(directory.glob("*.json")):
        try:
            with json_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"[WARN] Skipping {json_file.name}: JSON parse error – {exc}", file=sys.stderr)
            continue

        sequence_id = extract_sequence_id(data)
        if sequence_id is None:
            print(f"[WARN] Skipping {json_file.name}: missing 'sequence_id' field", file=sys.stderr)
            continue

        rows.append(
            {
                "sequence_id": str(sequence_id),
                "rep_id": "",  # intentionally blank for now
                "rep_json_path": "",  # intentionally blank for now
                "member_json_path": json_file.name,  # store file name only
            }
        )

    # Write index.csv
    index_path = directory / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Wrote {len(rows)} rows to {index_path.relative_to(Path.cwd())}")
    return len(rows)


def build_offset_file(csv_path: pathlib.Path, idx_path: pathlib.Path):
    offsets = array.array("Q", [0])            # first line starts at byte 0
    with csv_path.open("rb") as f:
        while f.readline():                    # read & discard line
            offsets.append(f.tell())           # record file pointer
    offsets.pop()            # last tell() == EOF → remove it
    idx_path.write_bytes(offsets.tobytes())    # 8-byte little-endian


def main() -> None:

    target_dir = Path(JSON_DIR).expanduser().resolve()
    try:
        build_index(target_dir)
        build_offset_file(target_dir / "index.csv", target_dir / "index.idx")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
