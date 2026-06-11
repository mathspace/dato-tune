from __future__ import annotations

import argparse
import logging
import sys
from typing import BinaryIO

import polars as pl
from rich.logging import RichHandler

from propagation_parameters_estimation.graph import (
    SOURCE_NODE_COL,
    SOURCE_SANA_TOPIC_ID_COL,
    TARGET_NODE_COL,
    TARGET_SANA_TOPIC_ID_COL,
    get_reachable_node_pairs,
)
from propagation_parameters_estimation.optimise import (
    DEFAULT_MAX_WLS_WEIGHT,
    DEFAULT_MIN_SHARED_STUDENTS,
    DEFAULT_MIN_SOURCE_VARIANCE,
    DEFAULT_UNSTABLE_SLOPE_MIN_DENOMINATOR,
    DEFAULT_UNSTABLE_SLOPE_MIN_SHARED_STUDENTS,
    DEFAULT_UNSTABLE_SLOPE_THRESHOLD,
    get_wls_parameter_result,
)
from propagation_parameters_estimation.updates import get_student_estimation_models

STUDENT_COL = "STUDENT_ID"
WINDOW_COL = "WINDOW_INDEX"
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
OPTIONAL_RESPONSE_READ_COLUMNS = [
    WINDOW_COL,
]
SKILL_LINK_READ_COLUMNS = [
    SOURCE_NODE_COL,
    TARGET_NODE_COL,
]
OPTIONAL_SKILL_LINK_READ_COLUMNS = [
    SOURCE_SANA_TOPIC_ID_COL,
    TARGET_SANA_TOPIC_ID_COL,
]

PARAMETER_SOURCE_NODE_COL = "source_node"
PARAMETER_TARGET_NODE_COL = "target_node"
PARAMETER_SOURCE_SANA_TOPIC_ID_COL = "source_sana_topic_id"
PARAMETER_TARGET_SANA_TOPIC_ID_COL = "target_sana_topic_id"
PARAMETER_OUTPUT_COLUMNS = [
    PARAMETER_SOURCE_NODE_COL,
    PARAMETER_SOURCE_SANA_TOPIC_ID_COL,
    PARAMETER_TARGET_NODE_COL,
    PARAMETER_TARGET_SANA_TOPIC_ID_COL,
    "L",
    "C",
    "default_reason",
    "shared_students",
    "invalid_input_count",
    "invalid_uncertainty_count",
    "source_variance",
    "weight_sum",
    "denominator",
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

    student_expr = pl.col(STUDENT_COL).cast(pl.Utf8)
    if WINDOW_COL in df.columns:
        student_expr = (
            pl.when(pl.col(WINDOW_COL).is_null())
            .then(student_expr)
            .otherwise(
                pl.concat_str(
                    [
                        pl.col(STUDENT_COL).cast(pl.Utf8),
                        pl.lit("::window="),
                        pl.col(WINDOW_COL).cast(pl.Utf8),
                    ]
                )
            )
        )

    prepared = df.select(
        student_expr.alias(CANONICAL_STUDENT_COL),
        pl.col(NODE_COL).cast(pl.Utf8).alias(CANONICAL_NODE_COL),
        pl.col(DIFFICULTY_COL).cast(pl.Float64).alias(CANONICAL_DIFFICULTY_COL),
        pl.col(DISCRIMINATION_COL).cast(pl.Float64).alias(CANONICAL_DISCRIMINATION_COL),
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
    min_shared_students: int = DEFAULT_MIN_SHARED_STUDENTS,
    min_source_variance: float = DEFAULT_MIN_SOURCE_VARIANCE,
    max_wls_weight: float = DEFAULT_MAX_WLS_WEIGHT,
    unstable_slope_threshold: float = DEFAULT_UNSTABLE_SLOPE_THRESHOLD,
    unstable_slope_min_shared_students: int = DEFAULT_UNSTABLE_SLOPE_MIN_SHARED_STUDENTS,
    unstable_slope_min_denominator: float = DEFAULT_UNSTABLE_SLOPE_MIN_DENOMINATOR,
    progress_interval: int = PAIR_PROGRESS_INTERVAL,
) -> pl.DataFrame:
    def log_stats_progress(students_processed: int, total_students: int):
        if should_log_progress(students_processed, total_students, progress_interval):
            logging.info(
                "Accumulated WLS stats for %s/%s students",
                f"{students_processed:,}",
                f"{total_students:,}",
            )

    def log_pair_progress(pair_fits_processed: int, total_pair_fits: int):
        if should_log_progress(pair_fits_processed, total_pair_fits, progress_interval):
            logging.info(
                "Fit WLS parameters for %s/%s directed node pairs",
                f"{pair_fits_processed:,}",
                f"{total_pair_fits:,}",
            )

    parameter_result = get_wls_parameter_result(
        student_estimation_models,
        allowed_node_pairs=allowed_node_pairs,
        min_shared_students=min_shared_students,
        min_source_variance=min_source_variance,
        max_wls_weight=max_wls_weight,
        unstable_slope_threshold=unstable_slope_threshold,
        unstable_slope_min_shared_students=unstable_slope_min_shared_students,
        unstable_slope_min_denominator=unstable_slope_min_denominator,
        stats_progress_callback=log_stats_progress,
        pair_progress_callback=log_pair_progress,
    )
    logging.info(
        "Used default WLS parameters for %s/%s directed node pairs",
        f"{parameter_result.default_count:,}",
        f"{len(parameter_result.parameter_rows):,}",
    )
    return pl.DataFrame(
        parameter_result.parameter_rows,
        schema={
            PARAMETER_SOURCE_NODE_COL: pl.Utf8,
            PARAMETER_TARGET_NODE_COL: pl.Utf8,
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


def get_skill_link_node_topic_ids(skill_links: pl.DataFrame) -> pl.DataFrame:
    topic_frames = []
    if SOURCE_SANA_TOPIC_ID_COL in skill_links.columns:
        topic_frames.append(
            skill_links.select(
                pl.col(SOURCE_NODE_COL).cast(pl.Utf8).alias("node"),
                pl.col(SOURCE_SANA_TOPIC_ID_COL)
                .cast(pl.Utf8)
                .alias("sana_topic_id"),
            )
        )
    if TARGET_SANA_TOPIC_ID_COL in skill_links.columns:
        topic_frames.append(
            skill_links.select(
                pl.col(TARGET_NODE_COL).cast(pl.Utf8).alias("node"),
                pl.col(TARGET_SANA_TOPIC_ID_COL)
                .cast(pl.Utf8)
                .alias("sana_topic_id"),
            )
        )

    if not topic_frames:
        return pl.DataFrame(
            schema={
                "node": pl.Utf8,
                "sana_topic_id": pl.Utf8,
            }
        )

    node_topic_ids = pl.concat(topic_frames).drop_nulls(["node", "sana_topic_id"])
    conflicts = (
        node_topic_ids.group_by("node")
        .agg(pl.col("sana_topic_id").n_unique().alias("topic_id_count"))
        .filter(pl.col("topic_id_count") > 1)
    )
    if conflicts.height:
        conflicting_nodes = conflicts.select("node").to_series().to_list()
        raise ValueError(
            "Conflicting SANA topic IDs for skill link nodes: "
            f"{conflicting_nodes[:10]}"
        )

    return node_topic_ids.unique("node", keep="first")


def add_skill_link_topic_ids(
    parameters: pl.DataFrame,
    skill_links: pl.DataFrame,
) -> pl.DataFrame:
    node_topic_ids = get_skill_link_node_topic_ids(skill_links)
    source_topic_ids = node_topic_ids.rename(
        {
            "node": PARAMETER_SOURCE_NODE_COL,
            "sana_topic_id": PARAMETER_SOURCE_SANA_TOPIC_ID_COL,
        }
    )
    target_topic_ids = node_topic_ids.rename(
        {
            "node": PARAMETER_TARGET_NODE_COL,
            "sana_topic_id": PARAMETER_TARGET_SANA_TOPIC_ID_COL,
        }
    )

    return (
        parameters.join(source_topic_ids, on=PARAMETER_SOURCE_NODE_COL, how="left")
        .join(target_topic_ids, on=PARAMETER_TARGET_NODE_COL, how="left")
        .select(PARAMETER_OUTPUT_COLUMNS)
    )


def estimate_propagation_parameters(
    responses: pl.DataFrame,
    *,
    skill_links: pl.DataFrame,
    curriculum_id: int | None = None,
    min_shared_students: int = DEFAULT_MIN_SHARED_STUDENTS,
    min_source_variance: float = DEFAULT_MIN_SOURCE_VARIANCE,
    max_wls_weight: float = DEFAULT_MAX_WLS_WEIGHT,
    unstable_slope_threshold: float = DEFAULT_UNSTABLE_SLOPE_THRESHOLD,
    unstable_slope_min_shared_students: int = DEFAULT_UNSTABLE_SLOPE_MIN_SHARED_STUDENTS,
    unstable_slope_min_denominator: float = DEFAULT_UNSTABLE_SLOPE_MIN_DENOMINATOR,
    progress_interval: int = PROGRESS_INTERVAL,
    pair_progress_interval: int = PAIR_PROGRESS_INTERVAL,
) -> pl.DataFrame:
    prepared_responses = prepare_response_data(responses, curriculum_id=curriculum_id)
    logging.info("Prepared %s responses", f"{prepared_responses.height:,}")
    logging.info(
        "Estimating ability models from %s responses", f"{prepared_responses.height:,}"
    )

    def log_student_estimation_progress(processed: int, students_seen: int):
        if should_log_progress(processed, prepared_responses.height, progress_interval):
            logging.info(
                "Estimated ability models from %s/%s responses across %s students",
                f"{processed:,}",
                f"{prepared_responses.height:,}",
                f"{students_seen:,}",
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
        f"{len(allowed_node_pairs):,}",
    )
    logging.info(
        "Fitting WLS parameters for %s students",
        f"{len(student_estimation_models):,}",
    )
    parameters = get_wls_parameters(
        student_estimation_models,
        allowed_node_pairs=allowed_node_pairs,
        min_shared_students=min_shared_students,
        min_source_variance=min_source_variance,
        max_wls_weight=max_wls_weight,
        unstable_slope_threshold=unstable_slope_threshold,
        unstable_slope_min_shared_students=unstable_slope_min_shared_students,
        unstable_slope_min_denominator=unstable_slope_min_denominator,
        progress_interval=pair_progress_interval,
    )
    return add_skill_link_topic_ids(parameters, skill_links)


def get_response_read_columns(curriculum_id: int | None = None) -> list[str]:
    columns = RESPONSE_READ_COLUMNS.copy()
    if curriculum_id is not None:
        columns.append(CURRICULUM_COL)
    return columns


def get_available_response_read_columns(
    available_columns: list[str], curriculum_id: int | None = None
) -> list[str]:
    columns = get_response_read_columns(curriculum_id)
    columns.extend(
        col for col in OPTIONAL_RESPONSE_READ_COLUMNS if col in available_columns
    )
    return columns


def get_available_skill_link_read_columns(available_columns: list[str]) -> list[str]:
    missing_cols = set(SKILL_LINK_READ_COLUMNS) - set(available_columns)
    if missing_cols:
        raise ValueError(f"Missing required skill link columns: {sorted(missing_cols)}")

    columns = SKILL_LINK_READ_COLUMNS.copy()
    columns.extend(
        col for col in OPTIONAL_SKILL_LINK_READ_COLUMNS if col in available_columns
    )
    return columns


def read_response_csv(infile: str, *, curriculum_id: int | None = None) -> pl.DataFrame:
    source: str | BinaryIO = sys.stdin.buffer if infile == "-" else infile
    if infile == "-":
        responses = pl.read_csv(source)
        return responses.select(
            get_available_response_read_columns(responses.columns, curriculum_id)
        )

    columns = get_available_response_read_columns(
        pl.read_csv(infile, n_rows=0).columns,
        curriculum_id,
    )
    return pl.read_csv(source, columns=columns).select(columns)


def read_response_csvs(
    infiles: list[str],
    *,
    curriculum_id: int | None = None,
) -> pl.DataFrame:
    if not infiles:
        raise ValueError("at least one response infile is required")
    responses = [
        read_response_csv(infile, curriculum_id=curriculum_id) for infile in infiles
    ]
    return pl.concat(responses, how="diagonal_relaxed")


def read_skill_links_csv(infile: str) -> pl.DataFrame:
    columns = get_available_skill_link_read_columns(
        pl.read_csv(infile, n_rows=0).columns
    )
    return pl.read_csv(infile, columns=columns).select(columns)


def write_parameters_csv(parameters: pl.DataFrame, outfile: str):
    if outfile == "-":
        sys.stdout.write(parameters.write_csv())
        return
    parameters.write_csv(outfile)


def run_estimate(args: argparse.Namespace):
    responses = read_response_csvs(
        [args.lantern_infile, args.mathspace_infile],
        curriculum_id=args.curriculum_id,
    )
    skill_links = read_skill_links_csv(args.skill_links_infile)
    parameters = estimate_propagation_parameters(
        responses,
        skill_links=skill_links,
        curriculum_id=args.curriculum_id,
        min_shared_students=args.min_shared_students,
        min_source_variance=args.min_source_variance,
        max_wls_weight=args.max_wls_weight,
        unstable_slope_threshold=args.unstable_slope_threshold,
        unstable_slope_min_shared_students=args.unstable_slope_min_shared_students,
        unstable_slope_min_denominator=args.unstable_slope_min_denominator,
    )
    write_parameters_csv(parameters, args.outfile)


def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate_parser = subparsers.add_parser("estimate")
    _ = estimate_parser.add_argument(
        "--lantern-infile",
        type=str,
        required=True,
        help="CSV file of Lantern responses.",
    )
    _ = estimate_parser.add_argument(
        "--mathspace-infile",
        type=str,
        required=True,
        help="CSV file of Mathspace responses.",
    )
    _ = estimate_parser.add_argument("--skill-links-infile", type=str, required=True)
    _ = estimate_parser.add_argument("--outfile", type=str, default="-")
    _ = estimate_parser.add_argument("--curriculum-id", type=int)
    _ = estimate_parser.add_argument(
        "--min-shared-students",
        type=int,
        default=DEFAULT_MIN_SHARED_STUDENTS,
    )
    _ = estimate_parser.add_argument(
        "--min-source-variance",
        type=float,
        default=DEFAULT_MIN_SOURCE_VARIANCE,
    )
    _ = estimate_parser.add_argument(
        "--max-wls-weight",
        type=float,
        default=DEFAULT_MAX_WLS_WEIGHT,
    )
    _ = estimate_parser.add_argument(
        "--unstable-slope-threshold",
        type=float,
        default=DEFAULT_UNSTABLE_SLOPE_THRESHOLD,
    )
    _ = estimate_parser.add_argument(
        "--unstable-slope-min-shared-students",
        type=int,
        default=DEFAULT_UNSTABLE_SLOPE_MIN_SHARED_STUDENTS,
    )
    _ = estimate_parser.add_argument(
        "--unstable-slope-min-denominator",
        type=float,
        default=DEFAULT_UNSTABLE_SLOPE_MIN_DENOMINATOR,
    )

    args = parser.parse_args()
    if args.command == "estimate":
        run_estimate(args)
    else:
        parser.error(f"Invalid command: {args.command}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
