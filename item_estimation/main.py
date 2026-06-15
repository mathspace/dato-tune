# pyright: reportAny=false
from __future__ import annotations

import argparse
import logging
import os
import re
import sys

from rich.logging import RichHandler
from configparser import ConfigParser, ExtendedInterpolation
from contextlib import nullcontext
from datetime import date, datetime
from typing import Literal

import polars as pl

logger = logging.getLogger(__name__)

from item_estimation.fetch import (
    fetch_lantern_repsonses_range,
    fetch_lantern_responses_windowed_batched,
    fetch_mathspace_responses_windowed_batched,
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


def run_fetch_lantern(
    outfile: str,
    curriculum_id: int,
    region: Literal["au", "us"],
    window_size_months: int | None = None,
    begin_date: date | None = None,
    end_date: date | None = None,
    max_windows: int | None = None,
    window_index: int | None = None,
):
    if window_size_months is not None:
        fetch_lantern_responses_windowed_batched(
            curriculum_id, region, window_size_months, outfile,
            max_windows=max_windows, window_index=window_index,
        )
    else:
        if begin_date is None or end_date is None:
            raise ValueError("begin_date and end_date are required when not using windowed mode")
        df = fetch_lantern_repsonses_range(curriculum_id, region, begin_date, end_date)
        with _output_file_context(outfile) as f:
            df.to_csv(f, index=False)


def run_fetch_mathspace(
    outfile: str,
    curriculum_id: int,
    region: Literal["au", "us"],
    window_size_months: int,
    max_windows: int | None = None,
    window_index: int | None = None,
):
    fetch_mathspace_responses_windowed_batched(
        curriculum_id, region, window_size_months, outfile,
        max_windows=max_windows, window_index=window_index,
    )


def run_inference(
    config: ConfigParser,
    outfile_suffix: str,
    curriculum_id: int,
    lantern_infile: str | None = None,
    mathspace_infile: str | None = None,
):
    student_sample_rate = config["inference"].getfloat("student_sample_rate", fallback=1.0)
    dfs = []

    if lantern_infile is not None:
        size_mb = os.path.getsize(lantern_infile) / 1_000_000
        logger.info(f"loading lantern file: {lantern_infile} ({size_mb:.1f}MB)")
        raw = pl.scan_csv(lantern_infile).collect()
        logger.info(f"  {len(raw):,} rows")
        dfs.append(preprocess_qa_df(raw, curriculum_id, add_default_values=False))

    if mathspace_infile is not None:
        size_mb = os.path.getsize(mathspace_infile) / 1_000_000
        logger.info(f"loading mathspace file: {mathspace_infile} ({size_mb:.1f}MB)")
        raw = pl.scan_csv(mathspace_infile).collect()
        logger.info(f"  {len(raw):,} rows")
        dfs.append(preprocess_qa_df(raw, curriculum_id, add_default_values=False, student_sample_rate=student_sample_rate))

    df = pl.concat(dfs, how="diagonal_relaxed")
    logger.info(f"  {len(df):,} rows after preprocessing")
    logger.info("starting inference...")
    run(config, df, outfile_suffix)


def _output_file_context(filename: str):
    return nullcontext(sys.stdout) if filename == "-" else open(filename, "w")


def _yyyy_mm_dd_date(x: str):
    return datetime.strptime(x, "%Y-%m-%d").date()


def _parse_window_size(window_size: str) -> int:
    """Parse window size string (e.g., '12m', '1y') and return months as integer."""
    match = re.match(r'^(\d+)([my])$', window_size.lower())
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid window size format: '{window_size}'. Expected format: <number>m (months) or <number>y (years), e.g., '12m' or '1y'"
        )

    value, unit = match.groups()
    months = int(value) * 12 if unit == 'y' else int(value)

    if months <= 0:
        raise argparse.ArgumentTypeError(f"Window size must be positive, got: {window_size}")

    return months


def main():
    config = ConfigParser(
        interpolation=ExtendedInterpolation(), default_section="common"
    )
    if not config.read("config.ini"):
        raise RuntimeError(
            "config.ini not found -- run `cp config.ini.example config.ini` to create it"
        )

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_common_fetch_args(p: argparse.ArgumentParser, window_required: bool):
        _ = p.add_argument("--outfile", type=str, required=True, help="Output CSV file path.")
        _ = p.add_argument("--curriculum-id", type=int, required=True)
        _ = p.add_argument(
            "--region",
            type=str,
            choices=["us", "au"],
            required=True,
            help="Region for Snowflake account: 'us' or 'au'",
        )
        _ = p.add_argument(
            "--window-size",
            "-w",
            type=_parse_window_size,
            required=window_required,
            help="Window size for sliding windows (e.g., '12m' for 12 months, '1y' for 1 year). Uses same stride as window size.",
        )
        window_group = p.add_mutually_exclusive_group()
        _ = window_group.add_argument(
            "--max-windows",
            type=int,
            default=None,
            help="Limit to this many windows (most recent first, e.g. --max-windows 3 fetches windows 0, 1, 2).",
        )
        _ = window_group.add_argument(
            "--window-index",
            type=int,
            default=None,
            help="Fetch only this specific window index (0 = most recent).",
        )

    fetch_lantern_parser = subparsers.add_parser("fetch-lantern")
    _add_common_fetch_args(fetch_lantern_parser, window_required=False)
    _ = fetch_lantern_parser.add_argument(
        "--begin-date",
        type=_yyyy_mm_dd_date,
        required=False,
        help="Start date (YYYY-MM-DD), required if not using --window-size",
    )
    _ = fetch_lantern_parser.add_argument(
        "--end-date",
        type=_yyyy_mm_dd_date,
        required=False,
        help="End date (YYYY-MM-DD), required if not using --window-size",
    )

    fetch_mathspace_parser = subparsers.add_parser("fetch-mathspace")
    _add_common_fetch_args(fetch_mathspace_parser, window_required=True)

    inference_parser = subparsers.add_parser("infer")
    _ = inference_parser.add_argument("--lantern-infile", type=str, default=None, help="CSV file of Lantern responses.")
    _ = inference_parser.add_argument("--mathspace-infile", type=str, default=None, help="CSV file of Mathspace responses. Student sampling applies to this file only.")
    _ = inference_parser.add_argument("--outfile-suffix", type=str, default="-")
    _ = inference_parser.add_argument("--curriculum-id", type=int, required=True)

    args = parser.parse_args()

    logfile = config["common"].get("logfile", None)
    if logfile and args.command.startswith("fetch-"):
        logfile = os.path.join(os.path.dirname(logfile), "fetch_" + os.path.basename(logfile))
    setup_logging(logfile)

    if args.command == "fetch-lantern":
        if args.window_size is None and (args.begin_date is None or args.end_date is None):
            parser.error("--begin-date and --end-date are required when not using --window-size")
        if args.window_size is not None and (args.begin_date is not None or args.end_date is not None):
            parser.error("--begin-date and --end-date cannot be used with --window-size")

        run_fetch_lantern(
            args.outfile,
            args.curriculum_id,
            args.region,
            args.window_size,
            args.begin_date,
            args.end_date,
            args.max_windows,
            args.window_index,
        )
    elif args.command == "fetch-mathspace":
        run_fetch_mathspace(args.outfile, args.curriculum_id, args.region, args.window_size, args.max_windows, args.window_index)
    elif args.command == "infer":
        if args.lantern_infile is None and args.mathspace_infile is None:
            parser.error("at least one of --lantern-infile or --mathspace-infile is required")
        run_inference(
            config,
            outfile_suffix=args.outfile_suffix,
            curriculum_id=args.curriculum_id,
            lantern_infile=args.lantern_infile,
            mathspace_infile=args.mathspace_infile,
        )
    else:
        parser.error(f"Invalid command: {args.command}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
