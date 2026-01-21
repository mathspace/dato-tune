# pyright: reportAny=false
from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from datetime import date, datetime
from typing import TextIO


def run_fetch_data(
    outfile: TextIO, curriculum_id: int, begin_date: date, end_date: date
):
    print(
        f"Fetching data for curriculum_id: {curriculum_id}, begin_date: {begin_date}, end_date: {end_date}"
    )


def run_validate_data(infile: TextIO, curriculum_id: int):
    print(f"Validating data for curriculum_id: {curriculum_id}")
    pass


def run_inference(infile: TextIO, outfile: TextIO, curriculum_id: int):
    print(f"Running inference for curriculum_id: {curriculum_id}")
    pass


def _input_file_context(filename: str):
    return nullcontext(sys.stdin) if filename == "-" else open(filename, "r")


def _output_file_context(filename: str):
    return nullcontext(sys.stdout) if filename == "-" else open(filename, "w")


def _yyyy_mm_dd_date(x: str):
    return datetime.strptime(x, "%Y-%m-%d").date()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch")
    _ = fetch_parser.add_argument("--outfile", type=str, default="-")
    _ = fetch_parser.add_argument("--curriculum-id", type=int, required=True)
    _ = fetch_parser.add_argument(
        "--begin-date",
        type=_yyyy_mm_dd_date,
        required=True,
    )
    _ = fetch_parser.add_argument(
        "--end-date",
        type=_yyyy_mm_dd_date,
        required=True,
    )

    validate_parser = subparsers.add_parser("validate")
    _ = validate_parser.add_argument("--infile", type=str, default="-")
    _ = validate_parser.add_argument("--curriculum-id", type=int, required=True)

    inference_parser = subparsers.add_parser("infer")
    _ = inference_parser.add_argument("--infile", type=str, default="-")
    _ = inference_parser.add_argument("--outfile", type=str, default="-")
    _ = inference_parser.add_argument("--curriculum-id", type=int, required=True)

    args = parser.parse_args()

    if args.command == "fetch":
        with _output_file_context(args.outfile) as outfile:
            run_fetch_data(outfile, args.curriculum_id, args.begin_date, args.end_date)
    elif args.command == "validate":
        with _input_file_context(args.infile) as infile:
            run_validate_data(infile, args.curriculum_id)
    elif args.command == "infer":
        with (
            _input_file_context(args.infile) as infile,
            _output_file_context(args.outfile) as outfile,
        ):
            run_inference(infile, outfile, args.curriculum_id)
    else:
        parser.error(f"Invalid command: {args.command}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
