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

from item_estimation.fetch import fetch_lantern_responses_from_snowflake
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
    outfile: TextIO, curriculum_id: int, region: str, begin_date: date, end_date: date
):
    df = fetch_lantern_responses_from_snowflake(
        curriculum_id, region, begin_date, end_date
    )
    df.to_csv(outfile)


def run_inference(
    config: ConfigParser, infile: TextIO, outfile: TextIO, curriculum_id: int
):
    df = preprocess_qa_df(pd.read_csv(infile), curriculum_id, add_default_values=False)
    run(config, df, outfile)


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
        "--begin-date",
        type=_yyyy_mm_dd_date,
        required=True,
    )
    _ = fetch_parser.add_argument(
        "--end-date",
        type=_yyyy_mm_dd_date,
        required=True,
    )

    inference_parser = subparsers.add_parser("infer")
    _ = inference_parser.add_argument("--infile", type=str, default="-")
    _ = inference_parser.add_argument("--outfile", type=str, default="-")
    _ = inference_parser.add_argument("--curriculum-id", type=int, required=True)

    args = parser.parse_args()

    if args.command == "fetch":
        with _output_file_context(args.outfile) as outfile:
            run_fetch_data(
                outfile, args.curriculum_id, args.region, args.begin_date, args.end_date
            )
    elif args.command == "infer":
        with (
            _input_file_context(args.infile) as infile,
            _output_file_context(args.outfile) as outfile,
        ):
            run_inference(config, infile, outfile, args.curriculum_id)
    else:
        parser.error(f"Invalid command: {args.command}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
