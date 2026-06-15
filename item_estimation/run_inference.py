import logging
import os
from configparser import ConfigParser
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import auc, roc_curve

import model_inference as mi
from model_inference import logistic_cdf
from utils import ColumnMapping


ITEM_LOW_RESPONSE_SUPPORT_THRESHOLD = 10
ITEM_LOW_STUDENT_SUPPORT_THRESHOLD = 10
ITEM_DISCRIMINATION_RUN_MAX_TOLERANCE = 1e-6
REQUIRED_INFERENCE_OPTIONS = [
    "result_folder",
    "n_iter",
    "tol",
    "infer_mastery",
    "infer_item",
    "tune_discrimination",
    "mastery_l2_penalty",
    "item_discrimination_l2_penalty",
    "discrimination_step_size",
    "show_graph",
    "random_seed",
    "min_obs",
    "item_low_response_support_threshold",
    "item_low_student_support_threshold",
    "mastery_correct_rate_tolerance",
    "item_freeze_correct_rate_tolerance",
    "split_ratio",
    "student_sample_rate",
]
REQUIRED_INFERENCE_OPTION_GROUPS = [
    ("granularity_sequence", "granularity_col"),
    ("item_freeze_min_response_support", "item_min_response_support"),
    ("item_freeze_min_student_support", "item_min_student_support"),
]


def has_config_value(inference_config, option: str) -> bool:
    return option in inference_config and inference_config.get(option).strip() != ""


def validate_required_inference_config(inference_config):
    missing = [
        option
        for option in REQUIRED_INFERENCE_OPTIONS
        if not has_config_value(inference_config, option)
    ]
    for option_group in REQUIRED_INFERENCE_OPTION_GROUPS:
        if not any(has_config_value(inference_config, option) for option in option_group):
            missing.append(" or ".join(option_group))
    if missing:
        raise ValueError(
            "missing required [inference] config option(s): "
            + ", ".join(missing)
        )


def run_mle(
    train_data: pl.DataFrame,
    granularity,
    infer_mastery=True,
    infer_item=True,
    tune_discrimination=False,
    frozen_item_ids=None,
    mastery_l2_penalty: float = 0.0,
    item_discrimination_l2_penalty: float = 0.0,
    discrimination_step_size: float | None = None,
    extra_cols_to_keep: list[str] | None = None,
    n_iter=30,
    tol=0.01,
):
    df = train_data
    frozen_item_ids = set(frozen_item_ids or [])
    # add default value as initial value of optimisation
    for col, default_value in zip(
        [ColumnMapping.mastery, ColumnMapping.difficulty, ColumnMapping.discrimination],
        [0.0, 0.0, 1.0],
    ):
        if col not in df.columns:
            df = df.with_columns(pl.lit(default_value).cast(pl.Float64).alias(col))

    # Select columns including window_col if provided
    cols_to_keep = [
        ColumnMapping.student_id,
        granularity,
        *(extra_cols_to_keep or []),
        ColumnMapping.estimate_question_id,
        ColumnMapping.score,
        ColumnMapping.difficulty,
        ColumnMapping.discrimination,
        ColumnMapping.mastery,
    ]
    cols_to_keep = list(dict.fromkeys(cols_to_keep))
    using_window_col = ColumnMapping.window_index in df.columns
    if using_window_col:
        cols_to_keep.append(ColumnMapping.window_index)

    df = df.select(cols_to_keep)
    df = df.with_columns(
        pl.col(ColumnMapping.estimate_question_id)
        .is_in(list(frozen_item_ids))
        .alias(mi.ITEM_PARAMETER_FROZEN_COL)
    )

    # Track initial item count for reporting
    total_items = df[ColumnMapping.estimate_question_id].n_unique()
    logging.info(f"Starting optimization with {total_items} unique items")
    if frozen_item_ids:
        logging.info(f"Freezing item parameters for {len(frozen_item_ids)} items")

    n_obs = len(df)
    likelihood = mi.total_likelihood(df)
    avg_likelihood = likelihood / mi.likelihood_denominator(df)

    estimation_tracking = [(0, likelihood, n_obs, avg_likelihood)]
    for it in range(n_iter):
        if infer_item:
            item_estimation_kwargs = {
                "item_discrimination_l2_penalty": item_discrimination_l2_penalty,
            }
            if discrimination_step_size is not None:
                item_estimation_kwargs["discrimination_step_size"] = discrimination_step_size
            df = mi.batch_item_estimation(
                df,
                tune_discrimination=tune_discrimination,
                frozen_item_ids=frozen_item_ids,
                **item_estimation_kwargs,
            )
            n_items = df[ColumnMapping.estimate_question_id].n_unique()
            item_pct = 100 * n_items / total_items
            logging.info(
                f"iteration: {it}, step: item estimation, items: {n_items}/{total_items} ({item_pct:.1f}%)"
            )

        if infer_mastery:
            df = mi.batch_mastery_estimation(
                df,
                granularity_col=granularity,
                using_window_col=using_window_col,
                mastery_l2_penalty=mastery_l2_penalty,
            )
            logging.info(f"iteration: {it}, step: mastery estimation, n_obs: {len(df):,}")

        n_obs = len(df)
        likelihood = mi.total_likelihood(df)
        avg_likelihood = likelihood / mi.likelihood_denominator(df)
        estimation_tracking.append((it + 1, likelihood, n_obs, avg_likelihood))
        logging.info(
            f"iteration: {it}, total likelihood: {likelihood}, n_obs: {n_obs}, avg: {avg_likelihood:.6f}"
        )

        if len(estimation_tracking) >= 3:
            prev_avg = estimation_tracking[-2][3]
            curr_avg = estimation_tracking[-1][3]
            relative_benefit = (curr_avg - prev_avg) / abs(prev_avg)
            logging.info(
                f"iteration: {it}, avg likelihood relative change: {relative_benefit:.4%}, tolerance: {tol:.4%}"
            )
            if 0 < relative_benefit < tol:
                logging.info(
                    f"optimisation stopped at iteration {it}, improvement: {relative_benefit:.4%}, tolerance: {tol:.4%}"
                )
                break

    df = df.with_columns(
        pl.Series(
            name=ColumnMapping.p_correct,
            values=mi.p_correct(
                df[ColumnMapping.mastery].to_numpy().astype(np.float64),
                df[ColumnMapping.difficulty].to_numpy().astype(np.float64),
                df[ColumnMapping.discrimination].to_numpy().astype(np.float64),
            ),
        )
    )
    estimation_tracking = pl.DataFrame(
        data=estimation_tracking,
        schema={"iter": pl.Int64, "likelihood": pl.Float64, "n_obs": pl.Int64, "avg_likelihood": pl.Float64},
        orient="row",
    )
    return estimation_tracking, df


def mle_track_plot(
    tracking: pl.DataFrame, title="Likelihood per iteration", file_name=None, display=True
):
    fig, ax = plt.subplots()
    p = sns.lineplot(x="iter", y="likelihood", data=tracking, ax=ax)
    p.set_title(title)
    if file_name:
        fig.savefig(file_name)
        logging.info(f"mle_track_plot: {title} saved as {file_name}")
    if display:
        plt.show()
    plt.close(fig)
    return


def roc_plot(df: pl.DataFrame, title="ROC curve", file_name=None, display=True):
    fpr, tpr, _ = roc_curve(
        df[ColumnMapping.score].to_numpy(),
        df[ColumnMapping.p_correct].to_numpy(),
    )
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    lw = 2
    ax.plot(
        fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    if file_name:
        fig.savefig(file_name)
        logging.info(f"roc_plot: {title} saved as {file_name}")
    if display:
        plt.show()
    plt.close(fig)
    return roc_auc


def brier_score(df: pl.DataFrame) -> float:
    if df.is_empty():
        return float("nan")
    return float(
        df.select(
            ((pl.col(ColumnMapping.p_correct) - pl.col(ColumnMapping.score)) ** 2)
            .mean()
            .alias("brier_score")
        ).item()
    )


def estimation_histogram(df: pl.DataFrame, title="Histogram", file_name=None, display=True):
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(18, 5))
    ax[0].hist(df[ColumnMapping.difficulty].to_numpy(), bins=30)
    ax[0].set_title("Estimated difficulty")

    ax[1].hist(df[ColumnMapping.discrimination].to_numpy(), bins=30)
    ax[1].set_title("Estimated discrimination index")

    ax[2].hist(df[ColumnMapping.mastery].to_numpy(), bins=30)
    ax[2].set_title("Estimated mastery")

    fig.suptitle(title, fontsize=14)
    if file_name:
        fig.savefig(file_name)
        logging.info(f"estimation_histogram: {title} saved as {file_name}")
    if display:
        plt.show()
    plt.close(fig)
    return


def add_item_warning_flags(
    estimated_difficulty: pl.DataFrame,
    low_response_support_threshold: int = ITEM_LOW_RESPONSE_SUPPORT_THRESHOLD,
    low_student_support_threshold: int = ITEM_LOW_STUDENT_SUPPORT_THRESHOLD,
) -> pl.DataFrame:
    if estimated_difficulty.is_empty():
        return estimated_difficulty.with_columns([
            pl.lit(False).alias("is_low_response_support"),
            pl.lit(False).alias("is_low_student_support"),
            pl.lit(False).alias("is_discrimination_at_run_max"),
            pl.lit(False).alias("has_item_warning"),
        ])

    max_discrimination = estimated_difficulty[ColumnMapping.discrimination].max()
    discrimination_run_max_threshold = (
        max_discrimination - ITEM_DISCRIMINATION_RUN_MAX_TOLERANCE
    )

    return estimated_difficulty.with_columns([
        (pl.col("n_response") < low_response_support_threshold).alias(
            "is_low_response_support"
        ),
        (pl.col("n_student") < low_student_support_threshold).alias(
            "is_low_student_support"
        ),
        (
            pl.col(ColumnMapping.discrimination) >= discrimination_run_max_threshold
        ).alias("is_discrimination_at_run_max"),
    ]).with_columns(
        (
            pl.col("is_low_response_support")
            | pl.col("is_low_student_support")
            | pl.col("is_discrimination_at_run_max")
        ).alias("has_item_warning")
    )


def get_result(
    train_result: pl.DataFrame,
    granularity,
    original_questions_dificulties: Dict[str, dict],
    file_path=None,
    outfile_suffix: str | None = None,
    using_window_col=False,
    item_low_response_support_threshold: int = ITEM_LOW_RESPONSE_SUPPORT_THRESHOLD,
    item_low_student_support_threshold: int = ITEM_LOW_STUDENT_SUPPORT_THRESHOLD,
):
    group_cols = [ColumnMapping.student_id, granularity]
    if using_window_col:
        group_cols.append(ColumnMapping.window_index)

    if mi.ITEM_PARAMETER_FROZEN_COL not in train_result.columns:
        train_result = train_result.with_columns(
            pl.lit(False).alias(mi.ITEM_PARAMETER_FROZEN_COL)
        )
    estimated_mastery = train_result.group_by(group_cols).agg([
        pl.col(ColumnMapping.mastery).mean(),
        pl.len().alias("n_response"),
        pl.col(ColumnMapping.score).sum().cast(pl.Int64).alias("n_correct"),
        (pl.len() - pl.col(ColumnMapping.score).sum()).cast(pl.Int64).alias("n_incorrect"),
        pl.col(ColumnMapping.score).mean().alias("observed_correct_rate"),
        pl.col(ColumnMapping.estimate_question_id).n_unique().alias("n_question"),
    ])

    estimated_difficulty = train_result.group_by(ColumnMapping.estimate_question_id).agg([
        pl.col(ColumnMapping.difficulty).mean(),
        pl.col(ColumnMapping.discrimination).mean(),
        pl.len().alias("n_response"),
        pl.col(ColumnMapping.score).sum().cast(pl.Int64).alias("n_correct"),
        (pl.len() - pl.col(ColumnMapping.score).sum()).cast(pl.Int64).alias("n_incorrect"),
        pl.col(ColumnMapping.score).mean().alias("observed_correct_rate"),
        pl.col(ColumnMapping.student_id).n_unique().alias("n_student"),
        pl.col(mi.ITEM_PARAMETER_FROZEN_COL).max().alias(mi.ITEM_PARAMETER_FROZEN_COL),
    ])
    estimated_difficulty = add_item_warning_flags(
        estimated_difficulty,
        low_response_support_threshold=item_low_response_support_threshold,
        low_student_support_threshold=item_low_student_support_threshold,
    )

    # Look up original difficulty by version_id (only exists for latest versions)
    estimated_difficulty = estimated_difficulty.with_columns(
        pl.col(ColumnMapping.estimate_question_id)
        .map_elements(
            lambda v_id: original_questions_dificulties.get(v_id, {}).get("difficulty"),
            return_dtype=pl.Float64,
        )
        .alias("OriginalDifficulty")
    )

    # Filter to only items with original difficulty (i.e., latest versions)
    estimated_difficulty_latest = estimated_difficulty.filter(
        pl.col("OriginalDifficulty").is_not_null()
    )
    estimated_difficulty_latest = estimated_difficulty_latest.with_columns(
        pl.Series(
            name="CalibratedDifficulty",
            values=logistic_cdf(
                estimated_difficulty_latest[ColumnMapping.difficulty].to_numpy().astype(np.float64)
            ),
        )
    )
    estimated_difficulty_latest = estimated_difficulty_latest.with_columns(
        (pl.col("CalibratedDifficulty") - pl.col("OriginalDifficulty"))
        .abs()
        .alias("DifficultiesDifference abs(Original-Calibrated)")
    )

    if outfile_suffix:
        with open(f"difficulties_{outfile_suffix}.csv", "w") as f:
            estimated_difficulty_latest.write_csv(f)
            logging.info("estimated difficulty saved to outfile")

    if file_path:
        mastery_file = os.path.join(file_path, "estimated_mastery.csv")
        difficulty_file = os.path.join(file_path, "estimated_item.csv")
        estimated_mastery.write_csv(mastery_file)
        estimated_difficulty_latest.write_csv(difficulty_file)
        logging.info(
            f"estimated mastery and difficulty are saved as files {mastery_file} and {difficulty_file}"
        )

    return estimated_mastery, estimated_difficulty


def calc_test_result(
    estimated_mastery: pl.DataFrame,
    estimated_difficulty: pl.DataFrame,
    test_data: pl.DataFrame,
    granularity,
    using_window_col=False,
):
    merge_cols = [ColumnMapping.student_id, granularity]
    if using_window_col:
        merge_cols.append(ColumnMapping.window_index)

    cols_to_drop = [
        c for c in [ColumnMapping.difficulty, ColumnMapping.discrimination]
        if c in test_data.columns
    ]
    df = (
        test_data.drop(cols_to_drop)
        .join(estimated_mastery, on=merge_cols, how="inner")
        .join(estimated_difficulty, on=ColumnMapping.estimate_question_id, how="inner")
    )
    df = df.with_columns(
        pl.Series(
            name=ColumnMapping.p_correct,
            values=mi.p_correct(
                df[ColumnMapping.mastery].to_numpy().astype(np.float64),
                df[ColumnMapping.difficulty].to_numpy().astype(np.float64),
                df[ColumnMapping.discrimination].to_numpy().astype(np.float64),
            ),
        )
    )
    logging.info(
        f"{np.round(df.shape[0] / test_data.shape[0], 2)} of test data has estimated mastery and difficulty"
    )
    return df


def get_questions_difficulties(df: pl.DataFrame) -> Dict[str, dict]:
    latest_versions = df.filter(
        pl.col(ColumnMapping.estimate_question_id) == pl.col(ColumnMapping.latest_question_version_id)
    )
    return {
        row[0]: {"public_id": row[1], "difficulty": row[2]}
        for row in latest_versions.select([
            ColumnMapping.estimate_question_id,
            ColumnMapping.question_public_id,
            ColumnMapping.difficulty,
        ]).unique().iter_rows()
    }


def correct_rate_range_from_tolerance(tolerance: float) -> tuple[float, float]:
    if not 0.0 <= tolerance <= 0.5:
        raise ValueError(
            f"correct rate tolerance must be between 0.0 and 0.5 inclusive, got {tolerance}"
        )
    return tolerance, 1.0 - tolerance


def set_frozen_item_difficulty_from_correct_rate(
    df: pl.DataFrame,
    frozen_item_ids: list,
    min_sigmoid_difficulty: float = 0.1,
    max_sigmoid_difficulty: float = 0.9,
) -> pl.DataFrame:
    if not frozen_item_ids:
        return df
    if not 0.0 < min_sigmoid_difficulty <= max_sigmoid_difficulty < 1.0:
        raise ValueError(
            "frozen item sigmoid difficulty bounds must satisfy "
            f"0.0 < min <= max < 1.0, got {min_sigmoid_difficulty}, {max_sigmoid_difficulty}"
        )
    frozen_item_ids = [str(item_id) for item_id in frozen_item_ids]
    df = df.with_columns(pl.col(ColumnMapping.estimate_question_id).cast(pl.Utf8))

    frozen_difficulties = (
        df.filter(pl.col(ColumnMapping.estimate_question_id).is_in(frozen_item_ids))
        .group_by(ColumnMapping.estimate_question_id)
        .agg(
            (1.0 - pl.col(ColumnMapping.score).mean())
            .clip(min_sigmoid_difficulty, max_sigmoid_difficulty)
            .alias("_frozen_sigmoid_difficulty")
        )
        .with_columns(
            (
                pl.col("_frozen_sigmoid_difficulty")
                / (1.0 - pl.col("_frozen_sigmoid_difficulty"))
            )
            .log()
            .alias("_frozen_difficulty")
        )
        .select(ColumnMapping.estimate_question_id, "_frozen_difficulty")
    )

    return df.join(
        frozen_difficulties,
        on=ColumnMapping.estimate_question_id,
        how="left",
    ).with_columns(
        pl.coalesce("_frozen_difficulty", ColumnMapping.difficulty)
        .alias(ColumnMapping.difficulty)
    ).drop("_frozen_difficulty")


def parse_granularity_sequence(inference_config) -> list[str]:
    sequence = inference_config.get("granularity_sequence", fallback="").strip()
    if not sequence:
        sequence = inference_config["granularity_col"].strip()

    granularities = [value.strip() for value in sequence.split(",")]
    if any(not value for value in granularities):
        raise ValueError(
            f"granularity_sequence must be a comma-separated list without empty entries, got {sequence!r}"
        )
    if len(set(granularities)) != len(granularities):
        raise ValueError(
            f"granularity_sequence must not contain duplicate granularities, got {sequence!r}"
        )
    return granularities


def parse_stage_int_sequence(
    inference_config,
    option_name: str,
    stage_count: int,
    default_value: int,
    fallback_option_name: str | None = None,
) -> list[int]:
    raw_value = inference_config.get(option_name, fallback="").strip()
    if not raw_value and fallback_option_name:
        raw_value = inference_config.get(fallback_option_name, fallback="").strip()
    if not raw_value:
        values = [default_value]
    else:
        values = [value.strip() for value in raw_value.split(",")]
        if any(not value for value in values):
            raise ValueError(
                f"{option_name} must be an integer or comma-separated integers without empty entries, "
                f"got {raw_value!r}"
            )
        try:
            values = [int(value) for value in values]
        except ValueError as exc:
            raise ValueError(
                f"{option_name} must be an integer or comma-separated integers, got {raw_value!r}"
            ) from exc

    if any(value < 0 for value in values):
        raise ValueError(f"{option_name} values must be non-negative, got {raw_value!r}")
    if len(values) == 1:
        return values * stage_count
    if len(values) != stage_count:
        raise ValueError(
            f"{option_name} must have one value or {stage_count} values for the granularity sequence, "
            f"got {len(values)} values"
        )
    return values


def resolve_granularity_columns(granularity_names: list[str], df: pl.DataFrame) -> list[str]:
    granularity_cols = []
    for granularity in granularity_names:
        granularity_col = getattr(ColumnMapping, granularity, None)
        if granularity_col is None:
            raise ValueError(f"granularity {granularity} is not a valid ColumnMapping attribute")
        if granularity_col not in df.columns:
            raise ValueError(
                f"granularity {granularity} maps to column {granularity_col}, "
                "but that column is not present in the input data"
            )
        granularity_cols.append(granularity_col)
    return granularity_cols


def _log_sanitization_stats(sanitization_stats: list[dict]):
    for stat in sanitization_stats:
        if stat["step"] == "item_parameter_freeze":
            logging.info(
                "  iteration %s, %s: %s items marked, %s by correct rate, "
                "%s by low support, %s rows remaining",
                stat["iteration"],
                stat["step"],
                f"{stat['items_marked']:,}",
                f"{stat.get('items_marked_by_correct_rate', 0):,}",
                f"{stat.get('items_marked_by_low_support', 0):,}",
                f"{stat['rows_after']:,}",
            )
        else:
            logging.info(
                "  iteration %s, %s: %s rows removed, %s rows remaining",
                stat["iteration"],
                stat["step"],
                f"{stat['rows_removed']:,}",
                f"{stat['rows_after']:,}",
            )


def _write_metrics(
    result_folder: Path,
    metrics_config: dict,
    df_estimation: pl.DataFrame,
    test_df_estimated: pl.DataFrame,
    test_df: pl.DataFrame,
    auc_train: float,
    brier_train: float,
    auc_test: float,
    brier_test: float,
):
    metrics_file = os.path.join(result_folder, "metrics.csv")
    pl.DataFrame([
        {
            "split": "train",
            "auc": auc_train,
            "brier_score": brier_train,
            "scored_rows": len(df_estimation),
            "total_rows": len(df_estimation),
            "coverage": 1.0,
            **metrics_config,
        },
        {
            "split": "test",
            "auc": auc_test,
            "brier_score": brier_test,
            "scored_rows": len(test_df_estimated),
            "total_rows": len(test_df),
            "coverage": len(test_df_estimated) / len(test_df) if len(test_df) else float("nan"),
            **metrics_config,
        },
    ]).write_csv(metrics_file)
    logging.info(f"metrics saved as {metrics_file}")


def _run_inference_stage(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    granularity_name: str,
    granularity_col: str,
    all_granularity_cols: list[str],
    original_difficulties: Dict[str, dict],
    result_folder: Path,
    outfile_suffix: str,
    metrics_config: dict,
    *,
    stage_index: int,
    stage_count: int,
    show_graph: bool,
    n_iter: int,
    tol: float,
    infer_mastery: bool,
    infer_item: bool,
    tune_discrimination: bool,
    mastery_l2_penalty: float,
    item_discrimination_l2_penalty: float,
    discrimination_step_size: float,
    min_obs: int,
    mastery_min_correct_rate: float,
    mastery_max_correct_rate: float,
    item_freeze_min_correct_rate: float,
    item_freeze_max_correct_rate: float,
    item_min_response_support: int,
    item_min_student_support: int,
    item_low_response_support_threshold: int,
    item_low_student_support_threshold: int,
    carried_frozen_item_ids: set,
    move_new_frozen_difficulty: bool,
    using_window_col: bool,
):
    result_folder.mkdir(exist_ok=True, parents=True)
    stage_name = f"stage_{stage_index}_{granularity_name}"
    group_cols = [ColumnMapping.student_id, granularity_col]
    if using_window_col:
        group_cols.append(ColumnMapping.window_index)

    logging.info(
        "stage %s/%s (%s): sanitizing training data: mastery min_obs >= %s, "
        "mastery correct rate %.2f-%.2f, item correct rate %.2f-%.2f "
        "item parameters frozen below %s responses/%s students",
        stage_index + 1,
        stage_count,
        granularity_name,
        min_obs,
        mastery_min_correct_rate,
        mastery_max_correct_rate,
        item_freeze_min_correct_rate,
        item_freeze_max_correct_rate,
        item_min_response_support,
        item_min_student_support,
    )
    train_df, stage_marked_frozen_item_ids, sanitization_stats = mi.sanitize_training_data(
        train_df,
        mastery_group_cols=group_cols,
        min_obs=min_obs,
        min_correct_rate=mastery_min_correct_rate,
        max_correct_rate=mastery_max_correct_rate,
        item_min_correct_rate=item_freeze_min_correct_rate,
        item_max_correct_rate=item_freeze_max_correct_rate,
        item_min_response=item_min_response_support,
        item_min_student=item_min_student_support,
    )
    _log_sanitization_stats(sanitization_stats)

    stage_marked_frozen_item_ids = set(stage_marked_frozen_item_ids)
    newly_added_frozen_item_ids = stage_marked_frozen_item_ids - carried_frozen_item_ids
    already_carried_marked_item_ids = stage_marked_frozen_item_ids & carried_frozen_item_ids
    if move_new_frozen_difficulty:
        train_df = set_frozen_item_difficulty_from_correct_rate(
            train_df,
            list(newly_added_frozen_item_ids),
        )

    frozen_item_ids = carried_frozen_item_ids | stage_marked_frozen_item_ids
    logging.info(
        "stage %s/%s (%s): train after sanitization: %s rows, %s carried frozen items, "
        "%s items marked by this stage, %s already carried and re-marked, "
        "%s newly added frozen items, %s total frozen items",
        stage_index + 1,
        stage_count,
        granularity_name,
        f"{len(train_df):,}",
        f"{len(carried_frozen_item_ids):,}",
        f"{len(stage_marked_frozen_item_ids):,}",
        f"{len(already_carried_marked_item_ids):,}",
        f"{len(newly_added_frozen_item_ids):,}",
        f"{len(frozen_item_ids):,}",
    )

    logging.info("stage %s/%s (%s): running MLE estimation...", stage_index + 1, stage_count, granularity_name)
    logging.info(f"  mastery_l2_penalty: {mastery_l2_penalty}")
    logging.info(f"  item_discrimination_l2_penalty: {item_discrimination_l2_penalty}")
    logging.info(f"  discrimination_step_size: {discrimination_step_size}")
    estimation_track, df_estimation = run_mle(
        train_df,
        granularity_col,
        infer_mastery=infer_mastery,
        infer_item=infer_item,
        tune_discrimination=tune_discrimination,
        frozen_item_ids=frozen_item_ids,
        mastery_l2_penalty=mastery_l2_penalty,
        item_discrimination_l2_penalty=item_discrimination_l2_penalty,
        discrimination_step_size=discrimination_step_size,
        extra_cols_to_keep=all_granularity_cols,
        n_iter=n_iter,
        tol=tol,
    )

    logging.info("stage %s/%s (%s): saving plots...", stage_index + 1, stage_count, granularity_name)
    mle_track_plot(
        estimation_track,
        file_name=os.path.join(result_folder, "inference_track.png"),
        display=show_graph,
    )
    estimation_histogram(
        df_estimation,
        file_name=os.path.join(result_folder, "inference_histogram.png"),
        display=show_graph,
    )

    logging.info("stage %s/%s (%s): saving results...", stage_index + 1, stage_count, granularity_name)
    trained_mastery, trained_difficulty = get_result(
        df_estimation,
        granularity_col,
        original_questions_dificulties=original_difficulties,
        file_path=result_folder,
        outfile_suffix=outfile_suffix if stage_index == stage_count - 1 else None,
        using_window_col=using_window_col,
        item_low_response_support_threshold=item_low_response_support_threshold,
        item_low_student_support_threshold=item_low_student_support_threshold,
    )

    logging.info("stage %s/%s (%s): calculating training ROC...", stage_index + 1, stage_count, granularity_name)
    auc_train = roc_plot(
        df_estimation,
        "training ROC",
        file_name=os.path.join(result_folder, "inference_roc_train.png"),
        display=show_graph,
    )
    brier_train = brier_score(df_estimation)
    logging.info(f"  training AUC: {auc_train:.4f}")
    logging.info(f"  training Brier score: {brier_train:.4f}")

    logging.info("stage %s/%s (%s): calculating test ROC...", stage_index + 1, stage_count, granularity_name)
    test_df_estimated = calc_test_result(
        trained_mastery,
        trained_difficulty,
        test_df,
        granularity_col,
        using_window_col=using_window_col,
    )
    auc_test = roc_plot(
        test_df_estimated,
        "testing ROC",
        file_name=os.path.join(result_folder, "inference_roc_test.png"),
        display=show_graph,
    )
    brier_test = brier_score(test_df_estimated)
    logging.info(f"  test AUC: {auc_test:.4f}")
    logging.info(f"  test Brier score: {brier_test:.4f}")

    _write_metrics(
        result_folder,
        {
            **metrics_config,
            "stage_index": stage_index,
            "stage_name": stage_name,
            "stage_granularity_col": granularity_name,
            "stage_item_freeze_min_response_support": item_min_response_support,
            "stage_item_freeze_min_student_support": item_min_student_support,
        },
        df_estimation,
        test_df_estimated,
        test_df,
        auc_train,
        brier_train,
        auc_test,
        brier_test,
    )

    return df_estimation, frozen_item_ids


def run(config: ConfigParser, df: pl.DataFrame, outfile_suffix: str):
    inference_config = config["inference"]
    validate_required_inference_config(inference_config)

    result_folder = Path(inference_config["result_folder"], outfile_suffix)
    result_folder.mkdir(exist_ok=True, parents=True)
    granularity_names = parse_granularity_sequence(inference_config)
    granularity_cols = resolve_granularity_columns(granularity_names, df)
    granularity_sequence = ",".join(granularity_names)
    n_iter = inference_config.getint("n_iter")
    tol = inference_config.getfloat("tol")
    infer_mastery = inference_config.getboolean("infer_mastery")
    infer_item = inference_config.getboolean("infer_item")
    tune_discrimination = inference_config.getboolean("tune_discrimination")
    mastery_l2_penalty = inference_config.getfloat("mastery_l2_penalty")
    if mastery_l2_penalty < 0:
        raise ValueError(f"mastery_l2_penalty must be non-negative, got {mastery_l2_penalty}")
    item_discrimination_l2_penalty = inference_config.getfloat("item_discrimination_l2_penalty")
    if item_discrimination_l2_penalty < 0:
        raise ValueError(
            f"item_discrimination_l2_penalty must be non-negative, got {item_discrimination_l2_penalty}"
        )
    discrimination_step_size = inference_config.getfloat(
        "discrimination_step_size"
    )
    if discrimination_step_size < 0:
        raise ValueError(
            f"discrimination_step_size must be non-negative, got {discrimination_step_size}"
        )
    show_graph = inference_config.getboolean("show_graph")
    random_seed = inference_config.getint("random_seed")
    min_obs = inference_config.getint("min_obs")
    item_low_response_support_threshold = inference_config.getint(
        "item_low_response_support_threshold"
    )
    item_low_student_support_threshold = inference_config.getint(
        "item_low_student_support_threshold"
    )
    item_freeze_min_response_support_by_stage = parse_stage_int_sequence(
        inference_config,
        "item_freeze_min_response_support",
        len(granularity_names),
        ITEM_LOW_RESPONSE_SUPPORT_THRESHOLD,
        fallback_option_name="item_min_response_support",
    )
    item_freeze_min_student_support_by_stage = parse_stage_int_sequence(
        inference_config,
        "item_freeze_min_student_support",
        len(granularity_names),
        ITEM_LOW_STUDENT_SUPPORT_THRESHOLD,
        fallback_option_name="item_min_student_support",
    )
    mastery_correct_rate_tolerance = inference_config.getfloat("mastery_correct_rate_tolerance")
    item_freeze_correct_rate_tolerance = inference_config.getfloat("item_freeze_correct_rate_tolerance")
    mastery_min_correct_rate, mastery_max_correct_rate = correct_rate_range_from_tolerance(
        mastery_correct_rate_tolerance
    )
    item_freeze_min_correct_rate, item_freeze_max_correct_rate = correct_rate_range_from_tolerance(
        item_freeze_correct_rate_tolerance
    )
    split_ratio = inference_config.getfloat("split_ratio")

    np.random.seed(random_seed)

    qa_history = df
    using_window_col = ColumnMapping.window_index in qa_history.columns

    logging.info("extracting question difficulties...")
    original_difficulties = get_questions_difficulties(qa_history)
    logging.info(f"  {len(original_difficulties):,} questions")

    logging.info(f"splitting train/test ({1 - split_ratio:.0%}/{split_ratio:.0%})...")
    train_df, test_df = mi.split_train_test_data_on_group(
        qa_history,
        [ColumnMapping.student_id, ColumnMapping.estimate_question_id],
        ratio=split_ratio,
    )
    logging.info(f"  train: {len(train_df):,} rows, test: {len(test_df):,} rows")

    metrics_config = {
        "outfile_suffix": outfile_suffix,
        "granularity_col": granularity_names[-1],
        "granularity_sequence": granularity_sequence,
        "n_iter": n_iter,
        "tol": tol,
        "infer_mastery": infer_mastery,
        "infer_item": infer_item,
        "tune_discrimination": tune_discrimination,
        "mastery_l2_penalty": mastery_l2_penalty,
        "item_discrimination_l2_penalty": item_discrimination_l2_penalty,
        "discrimination_step_size": discrimination_step_size,
        "min_obs": min_obs,
        "item_freeze_min_response_support": ",".join(
            str(value) for value in item_freeze_min_response_support_by_stage
        ),
        "item_freeze_min_student_support": ",".join(
            str(value) for value in item_freeze_min_student_support_by_stage
        ),
        "item_low_response_support_threshold": item_low_response_support_threshold,
        "item_low_student_support_threshold": item_low_student_support_threshold,
        "mastery_correct_rate_tolerance": mastery_correct_rate_tolerance,
        "item_freeze_correct_rate_tolerance": item_freeze_correct_rate_tolerance,
        "split_ratio": split_ratio,
        "random_seed": random_seed,
        "student_sample_rate": inference_config.getfloat("student_sample_rate"),
    }

    logging.info("granularity sequence: %s", granularity_sequence)
    stage_train_df = train_df
    frozen_item_ids = set()
    for stage_index, (granularity_name, granularity_col) in enumerate(
        zip(granularity_names, granularity_cols)
    ):
        is_final_stage = stage_index == len(granularity_names) - 1
        stage_name = f"stage_{stage_index}_{granularity_name}"
        stage_result_folder = (
            result_folder
            if is_final_stage
            else result_folder / stage_name
        )
        stage_train_df, frozen_item_ids = _run_inference_stage(
            stage_train_df,
            test_df,
            granularity_name,
            granularity_col,
            granularity_cols,
            original_difficulties,
            stage_result_folder,
            outfile_suffix,
            metrics_config,
            stage_index=stage_index,
            stage_count=len(granularity_names),
            show_graph=show_graph,
            n_iter=n_iter,
            tol=tol,
            infer_mastery=infer_mastery,
            infer_item=infer_item,
            tune_discrimination=tune_discrimination,
            mastery_l2_penalty=mastery_l2_penalty,
            item_discrimination_l2_penalty=item_discrimination_l2_penalty,
            discrimination_step_size=discrimination_step_size,
            min_obs=min_obs,
            mastery_min_correct_rate=mastery_min_correct_rate,
            mastery_max_correct_rate=mastery_max_correct_rate,
            item_freeze_min_correct_rate=item_freeze_min_correct_rate,
            item_freeze_max_correct_rate=item_freeze_max_correct_rate,
            item_min_response_support=item_freeze_min_response_support_by_stage[stage_index],
            item_min_student_support=item_freeze_min_student_support_by_stage[stage_index],
            item_low_response_support_threshold=item_low_response_support_threshold,
            item_low_student_support_threshold=item_low_student_support_threshold,
            carried_frozen_item_ids=frozen_item_ids,
            move_new_frozen_difficulty=stage_index == 0,
            using_window_col=using_window_col,
        )

    return
