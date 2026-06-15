from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}$"
)
MATHSPACE_PATTERN = re.compile(r"^\d+:\d+$")


def infer_source(question_version_id: str) -> str:
    if UUID_PATTERN.fullmatch(question_version_id):
        return "lantern"
    if MATHSPACE_PATTERN.fullmatch(question_version_id):
        return "mathspace"
    raise ValueError(f"Unsupported QUESTION_VERSION_ID format: {question_version_id!r}")


def build_fieldnames(fieldnames: list[str]) -> list[str]:
    if "QUESTION_VERSION_ID" not in fieldnames:
        raise ValueError("Missing required column: QUESTION_VERSION_ID")

    output_fieldnames = [name for name in fieldnames if name != "SOURCE"]
    question_version_index = output_fieldnames.index("QUESTION_VERSION_ID") + 1
    output_fieldnames.insert(question_version_index, "SOURCE")
    return output_fieldnames


def add_source_column(input_path: Path, output_path: Path) -> None:
    with input_path.open(newline="") as infile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None:
            raise ValueError("Input CSV is missing a header row")

        fieldnames = build_fieldnames(reader.fieldnames)
        rows: list[dict[str, str]] = []

        for row_number, row in enumerate(reader, start=2):
            question_version_id = row.get("QUESTION_VERSION_ID")
            if not question_version_id:
                raise ValueError(f"Missing QUESTION_VERSION_ID at row {row_number}")

            normalized_row = {name: row.get(name, "") for name in fieldnames}
            normalized_row["SOURCE"] = infer_source(question_version_id)
            rows.append(normalized_row)

    with output_path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a SOURCE column based on QUESTION_VERSION_ID."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="stage/estimated_item.csv",
        help="Path to the input CSV. Defaults to stage/estimated_item.csv.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output CSV path. Defaults to overwriting the input file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output) if args.output else input_path

    add_source_column(input_path, output_path)
    print(f"Wrote SOURCE column to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
