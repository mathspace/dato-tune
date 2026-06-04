from __future__ import annotations

import argparse
import logging
import sys
from typing import BinaryIO

import polars as pl
from rich.logging import RichHandler

from propagation_parameters_estimation.graph import (
    SOURCE_NODE_COL,
    TARGET_NODE_COL,
    get_reachable_node_pairs,
)
from propagation_parameters_estimation.optimise import get_wls_parameter_result
from propagation_parameters_estimation.updates import get_student_estimation_models

STUDENT_COL = "STUDENT_ID"
NODE_COL = "SKILL_ID"
DIFFICULTY_COL = "COLD_START_DIFFICULTY"
DISCRIMINATION_COL = "DISCRIMINATION"
RESULT_COL = "RESULT"
CURRICULUM_COL = "CURRICULUM_ID"

CANONICAL_STUDENT_COL = "student_id"
CANONICAL_NODE_COL = "node"
CANONICAL_DIFFICULTY_COL = "difficulty"
CANONICAL_DISCRIMINATION_COL = "discrimination"
CANONICAL_RESPONSE_COL = "response"

REQUIRED_RESPONSE_COLUMNS = {
    STUDENT_COL,
    NODE_COL,
    DIFFICULTY_COL,
    DISCRIMINATION_COL,
    RESULT_COL,
}

RESPONSE_READ_COLUMNS = [
    STUDENT_COL,
    NODE_COL,
    DIFFICULTY_COL,
    DISCRIMINATION_COL,
    RESULT_COL,
]
SKILL_LINK_READ_COLUMNS = [
    SOURCE_NODE_COL,
    TARGET_NODE_COL,
]

PROGRESS_INTERVAL = 100_000
PAIR_PROGRESS_INTERVAL = 10_000


def setup_logging():
    logging.captureWarnings(True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(RichHandler(rich_tracebacks=True))


def should_log_progress(processed: int, total: int, interval: int) -> bool:
    return interval > 0 and (processed == total or processed % interval == 0)


def prepare_response_data(
    responses: pl.DataFrame,
    *,
    curriculum_id: int | None = None,
) -> pl.DataFrame:
    missing_cols = REQUIRED_RESPONSE_COLUMNS - set(responses.columns)
    if missing_cols:
        raise ValueError(f"Missing required response columns: {sorted(missing_cols)}")

    df = responses
    if curriculum_id is not None:
        if CURRICULUM_COL not in df.columns:
            raise ValueError(f"Missing required response column: {CURRICULUM_COL}")
        df = df.filter(pl.col(CURRICULUM_COL) == curriculum_id)

    prepared = df.select(
        pl.col(STUDENT_COL).cast(pl.Utf8).alias(CANONICAL_STUDENT_COL),
        pl.col(NODE_COL).cast(pl.Utf8).alias(CANONICAL_NODE_COL),
        pl.col(DIFFICULTY_COL).cast(pl.Float64).alias(CANONICAL_DIFFICULTY_COL),
        pl.col(DISCRIMINATION_COL)
        .cast(pl.Float64)
        .alias(CANONICAL_DISCRIMINATION_COL),
        (pl.col(RESULT_COL).cast(pl.Utf8).str.to_uppercase() == "CORRECT")
        .cast(pl.Int8)
        .alias(CANONICAL_RESPONSE_COL),
    )
    prepared = prepared.drop_nulls(
        [
            CANONICAL_STUDENT_COL,
            CANONICAL_NODE_COL,
            CANONICAL_DIFFICULTY_COL,
            CANONICAL_DISCRIMINATION_COL,
            CANONICAL_RESPONSE_COL,
        ]
    ).filter(
        pl.col(CANONICAL_DIFFICULTY_COL).is_finite()
        & pl.col(CANONICAL_DISCRIMINATION_COL).is_finite()
        & (pl.col(CANONICAL_DISCRIMINATION_COL) > 0.0)
    )

    return prepared.select(
        CANONICAL_STUDENT_COL,
        CANONICAL_NODE_COL,
        CANONICAL_DIFFICULTY_COL,
        CANONICAL_DISCRIMINATION_COL,
        CANONICAL_RESPONSE_COL,
    )


def get_wls_parameters(
    student_estimation_models: dict[str, dict[str, dict[str, float]]],
    *,
    allowed_node_pairs: set[tuple[str, str]],
    min_shared_students: int = 2,
    min_source_variance: float = 1e-8,
    progress_interval: int = PAIR_PROGRESS_INTERVAL,
) -> pl.DataFrame:
    def log_stats_progress(students_processed: int, total_students: int):
        if should_log_progress(students_processed, total_students, progress_interval):
            logging.info(
                "Accumulated WLS stats for %s/%s students",
                students_processed,
                total_students,
            )

    def log_pair_progress(pair_fits_processed: int, total_pair_fits: int):
        if should_log_progress(
            pair_fits_processed, total_pair_fits, progress_interval
        ):
            logging.info(
                "Fit WLS parameters for %s/%s directed node pairs",
                pair_fits_processed,
                total_pair_fits,
            )

    parameter_result = get_wls_parameter_result(
        student_estimation_models,
        allowed_node_pairs=allowed_node_pairs,
        min_shared_students=min_shared_students,
        min_source_variance=min_source_variance,
        stats_progress_callback=log_stats_progress,
        pair_progress_callback=log_pair_progress,
    )
    logging.info(
        "Used default WLS parameters for %s/%s directed node pairs",
        parameter_result.default_count,
        len(parameter_result.parameter_rows),
    )
    return pl.DataFrame(
        parameter_result.parameter_rows,
        schema={
            "source_node": pl.Utf8,
            "target_node": pl.Utf8,
            "L": pl.Float64,
            "C": pl.Float64,
            "default_reason": pl.Utf8,
            "shared_students": pl.UInt32,
            "invalid_input_count": pl.UInt32,
            "invalid_uncertainty_count": pl.UInt32,
            "source_variance": pl.Float64,
            "weight_sum": pl.Float64,
            "denominator": pl.Float64,
        },
        orient="row",
    )


def estimate_propagation_parameters(
    responses: pl.DataFrame,
    *,
    skill_links: pl.DataFrame,
    curriculum_id: int | None = None,
    min_shared_students: int = 2,
    min_source_variance: float = 1e-8,
    progress_interval: int = PROGRESS_INTERVAL,
    pair_progress_interval: int = PAIR_PROGRESS_INTERVAL,
) -> pl.DataFrame:
    prepared_responses = prepare_response_data(responses, curriculum_id=curriculum_id)
    logging.info("Prepared %s responses", prepared_responses.height)
    logging.info(
        "Estimating ability models from %s responses", prepared_responses.height
    )

    def log_student_estimation_progress(processed: int, students_seen: int):
        if should_log_progress(
            processed, prepared_responses.height, progress_interval
        ):
            logging.info(
                "Estimated ability models from %s/%s responses across %s students",
                processed,
                prepared_responses.height,
                students_seen,
            )

    student_estimation_models = get_student_estimation_models(
        prepared_responses.iter_rows(),
        progress_callback=log_student_estimation_progress,
        progress_interval=progress_interval,
        total_responses=prepared_responses.height,
    )
    allowed_node_pairs = get_reachable_node_pairs(skill_links)
    logging.info(
        "Prepared %s reachable directed node pairs from skill links",
        len(allowed_node_pairs),
    )
    logging.info("Fitting WLS parameters for %s students", len(student_estimation_models))
    return get_wls_parameters(
        student_estimation_models,
        allowed_node_pairs=allowed_node_pairs,
        min_shared_students=min_shared_students,
        min_source_variance=min_source_variance,
        progress_interval=pair_progress_interval,
    )


def get_response_read_columns(curriculum_id: int | None = None) -> list[str]:
    columns = RESPONSE_READ_COLUMNS.copy()
    if curriculum_id is not None:
        columns.append(CURRICULUM_COL)
    return columns


def read_response_csv(
    infile: str, *, curriculum_id: int | None = None
) -> pl.DataFrame:
    source: str | BinaryIO = sys.stdin.buffer if infile == "-" else infile
    return pl.read_csv(source, columns=get_response_read_columns(curriculum_id))


def read_skill_links_csv(infile: str) -> pl.DataFrame:
    return pl.read_csv(infile, columns=SKILL_LINK_READ_COLUMNS)


def write_parameters_csv(parameters: pl.DataFrame, outfile: str):
    if outfile == "-":
        sys.stdout.write(parameters.write_csv())
        return
    parameters.write_csv(outfile)


def run_estimate(args: argparse.Namespace):
    responses = read_response_csv(args.infile, curriculum_id=args.curriculum_id)
    skill_links = read_skill_links_csv(args.skill_links_infile)
    parameters = estimate_propagation_parameters(
        responses,
        skill_links=skill_links,
        curriculum_id=args.curriculum_id,
        min_shared_students=args.min_shared_students,
        min_source_variance=args.min_source_variance,
    )
    write_parameters_csv(parameters, args.outfile)


def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate_parser = subparsers.add_parser("estimate")
    _ = estimate_parser.add_argument("--infile", type=str, default="-")
    _ = estimate_parser.add_argument("--skill-links-infile", type=str, required=True)
    _ = estimate_parser.add_argument("--outfile", type=str, default="-")
    _ = estimate_parser.add_argument("--curriculum-id", type=int)
    _ = estimate_parser.add_argument("--min-shared-students", type=int, default=2)
    _ = estimate_parser.add_argument("--min-source-variance", type=float, default=1e-8)

    args = parser.parse_args()
    if args.command == "estimate":
        run_estimate(args)
    else:
        parser.error(f"Invalid command: {args.command}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
