# pyright: reportAny=false
from __future__ import annotations

import argparse
import logging
import sys

from rich.logging import RichHandler
from configparser import ConfigParser, ExtendedInterpolation
from contextlib import nullcontext
from datetime import date, datetime
from typing import TextIO

import pandas as pd

from item_estimation.fetch import (
    fetch_lantern_repsonses_range,
    fetch_lantern_responses_windowed,
)
from item_estimation.load_data import preprocess_qa_df
from item_estimation.run_inference import run


def setup_logging(logfile: str | None = None):
    logging.captureWarnings(True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    stderr_handler = RichHandler(rich_tracebacks=True)
    stderr_handler.setLevel(logging.INFO)
    logger.addHandler(stderr_handler)

    if logfile:
        logging.info(f"Logging to file: {logfile}")
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
        )
        logger.addHandler(file_handler)


def run_fetch_data(
    outfile: TextIO,
    curriculum_id: int,
    region: str,
    windowed: bool,
    begin_date: date | None = None,
    end_date: date | None = None,
):
    if windowed:
        df = fetch_lantern_responses_windowed(curriculum_id, region)
    else:
        if begin_date is None or end_date is None:
            raise ValueError("begin_date and end_date are required when not using windowed mode")
        df = fetch_lantern_repsonses_range(curriculum_id, region, begin_date, end_date)
    df.to_csv(outfile, index=False)


def run_inference(
    config: ConfigParser, infile: TextIO, outfile_suffix: str, curriculum_id: int
):
    df = preprocess_qa_df(pd.read_csv(infile), curriculum_id, add_default_values=False)
    run(config, df, outfile_suffix)

def _input_file_context(filename: str):
    return nullcontext(sys.stdin) if filename == "-" else open(filename, "r")


def _output_file_context(filename: str):
    return nullcontext(sys.stdout) if filename == "-" else open(filename, "w")


def _yyyy_mm_dd_date(x: str):
    return datetime.strptime(x, "%Y-%m-%d").date()


def main():
    config = ConfigParser(
        interpolation=ExtendedInterpolation(), default_section="common"
    )
    if not config.read("config.ini"):
        raise RuntimeError(
            "config.ini not found -- run `cp config.ini.example config.ini` to create it"
        )

    setup_logging(config["common"].get("logfile", None))

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch")
    _ = fetch_parser.add_argument("--outfile", type=str, default="-")
    _ = fetch_parser.add_argument("--curriculum-id", type=int, required=True)
    _ = fetch_parser.add_argument(
        "--region",
        type=str,
        choices=["us", "au"],
        required=True,
        help="Region for Snowflake account: 'us' or 'au'",
    )
    _ = fetch_parser.add_argument(
        "--windowed",
        "-w",
        action="store_true",
        help="Use windowed mode (sliding 12-month windows with 6-month stride)",
    )
    _ = fetch_parser.add_argument(
        "--begin-date",
        type=_yyyy_mm_dd_date,
        required=False,
        help="Start date (YYYY-MM-DD), required if not using --windowed",
    )
    _ = fetch_parser.add_argument(
        "--end-date",
        type=_yyyy_mm_dd_date,
        required=False,
        help="End date (YYYY-MM-DD), required if not using --windowed",
    )

    inference_parser = subparsers.add_parser("infer")
    _ = inference_parser.add_argument("--infile", type=str, default="-")
    _ = inference_parser.add_argument("--outfile", type=str, default="-")
    _ = inference_parser.add_argument("--outfile-suffix", type=str, default="-")
    _ = inference_parser.add_argument("--curriculum-id", type=int, required=True)

    args = parser.parse_args()

    if args.command == "fetch":
        if not args.windowed and (args.begin_date is None or args.end_date is None):
            parser.error("--begin-date and --end-date are required when not using --windowed")
        if args.windowed and (args.begin_date is not None or args.end_date is not None):
            parser.error("--begin-date and --end-date cannot be used with --windowed")

        with _output_file_context(args.outfile) as outfile:
            run_fetch_data(
                outfile,
                args.curriculum_id,
                args.region,
                args.windowed,
                args.begin_date,
                args.end_date,
            )
    elif args.command == "infer":
        with (
            _input_file_context(args.infile) as infile,
        ):
            run_inference(config, infile, outfile_suffix=args.outfile_suffix, curriculum_id=args.curriculum_id)
    else:
        parser.error(f"Invalid command: {args.command}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
