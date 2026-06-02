import math

import numpy as np
import polars as pl
from numba import float64, njit, vectorize
from scipy import optimize

from utils import ColumnMapping


ITEM_PARAMETER_FROZEN_COL = "is_item_parameter_frozen"


@vectorize([float64(float64)])
def logit(p):
    return math.log(p / (1 - p))


@vectorize([float64(float64)])
def logistic_cdf(x):
    return 1 / (1 + math.exp(-x))


@vectorize([float64(float64)])
def logistic_sf(x):
    return 1 / (1 + math.exp(x))


@vectorize([float64(float64)])
def logistic_logsf(x):
    return math.log(logistic_sf(x))


@vectorize([float64(float64)])
def logistic_logcdf(x):
    return math.log(logistic_cdf(x))


@vectorize([float64(float64, float64, float64)])
def p_correct(m, d, v):
    return logistic_cdf(v * (m - d))


@njit([float64(float64[:], float64[:], float64[:], float64[:])])
def likelihood(m, d, v, r):
    return (logistic_logcdf(v * (m - d)) * r).sum() + (
        logistic_logsf(v * (m - d)) * (1 - r)
    ).sum()


## variations of likelihood, depending on if any of the variables are float instead of vector
@njit([float64(float64, float64[:], float64[:], float64[:])])
def likelihood_mastery(mastery, d, v, r):
    m = np.repeat(mastery, len(r))
    return likelihood(m, d, v, r)


@njit([float64(float64[:], float64, float64, float64[:])])
def likelihood_item(m, difficulty, discrimination, r):
    d = np.repeat(difficulty, len(r))
    v = np.repeat(discrimination, len(r))
    return likelihood(m, d, v, r)


## Numba estimation function
def estimate_mastery(d, v, r, **kwargs):
    mastery_l2_penalty = kwargs.pop("mastery_l2_penalty", 0.0)
    if mastery_l2_penalty < 0:
        raise ValueError(f"mastery_l2_penalty must be non-negative, got {mastery_l2_penalty}")

    if mastery_l2_penalty == 0.0:
        def h(m):
            return -likelihood_mastery(m, d, v, r)
    else:
        def h(m):
            return -likelihood_mastery(m, d, v, r) + mastery_l2_penalty * (m ** 2)

    return optimize.minimize_scalar(h, **kwargs)


def estimate_item(m, r, **kwargs):
    item_discrimination_l2_penalty = kwargs.pop("item_discrimination_l2_penalty", 0.0)
    if item_discrimination_l2_penalty < 0:
        raise ValueError(
            f"item_discrimination_l2_penalty must be non-negative, got {item_discrimination_l2_penalty}"
        )

    if item_discrimination_l2_penalty == 0.0:
        def h(item_params):
            d, v = item_params
            return -likelihood_item(m, d, v, r)
    else:
        def h(item_params):
            d, v = item_params
            return (
                -likelihood_item(m, d, v, r)
                + item_discrimination_l2_penalty * ((v - 1.0) ** 2)
            )

    if "x0" not in kwargs.keys():
        kwargs["x0"] = [0.0, 1.0]
    return optimize.minimize(h, **kwargs)


## Functions to run estimation on a data set


def split_train_test_data_on_group(df: pl.DataFrame, group_cols: list, ratio: float = 0.3):
    # Sort groups for deterministic ordering with np.random.seed
    group_counts = (
        df.select(group_cols)
        .group_by(group_cols)
        .len()
        .sort(group_cols)
    )
    n = len(group_counts)
    idx_test = np.random.rand(n) <= ratio
    idx_train = ~idx_test
    train_groups = group_counts.filter(pl.Series(idx_train)).select(group_cols)
    test_groups = group_counts.filter(pl.Series(idx_test)).select(group_cols)
    train_df = df.join(train_groups, on=group_cols, how="inner")
    test_df = df.join(test_groups, on=group_cols, how="inner")
    return train_df, test_df


def remove_groups_with_insufficient_data(df: pl.DataFrame, group_cols: list, min_obs: int):
    sufficient = (
        df.group_by(group_cols)
        .len()
        .filter(pl.col("len") >= min_obs)
        .select(group_cols)
    )
    return df.join(sufficient, on=group_cols, how="inner")


def remove_groups_outside_correct_rate_range(
    df: pl.DataFrame,
    group_cols: list,
    min_correct_rate: float = 0.1,
    max_correct_rate: float = 0.9,
):
    in_range = (
        df.group_by(group_cols)
        .agg(pl.col(ColumnMapping.score).mean().alias("observed_correct_rate"))
        .filter(
            pl.col("observed_correct_rate").is_between(
                min_correct_rate,
                max_correct_rate,
                closed="both",
            )
        )
        .select(group_cols)
    )
    return df.join(in_range, on=group_cols, how="inner")


def get_groups_outside_correct_rate_range(
    df: pl.DataFrame,
    group_cols: list,
    min_correct_rate: float = 0.1,
    max_correct_rate: float = 0.9,
) -> pl.DataFrame:
    return (
        df.group_by(group_cols)
        .agg(pl.col(ColumnMapping.score).mean().alias("observed_correct_rate"))
        .filter(
            ~pl.col("observed_correct_rate").is_between(
                min_correct_rate,
                max_correct_rate,
                closed="both",
            )
        )
        .select(group_cols)
    )


def get_frozen_item_ids_by_correct_rate(
    df: pl.DataFrame,
    min_correct_rate: float = 0.1,
    max_correct_rate: float = 0.9,
) -> list:
    if df.is_empty():
        return []
    return (
        get_groups_outside_correct_rate_range(
            df,
            [ColumnMapping.estimate_question_id],
            min_correct_rate,
            max_correct_rate,
        )
        .get_column(ColumnMapping.estimate_question_id)
        .to_list()
    )


def get_frozen_item_ids_by_support(
    df: pl.DataFrame,
    min_response: int | None = None,
    min_student: int | None = None,
) -> list:
    if df.is_empty() or (min_response is None and min_student is None):
        return []

    filters = []
    if min_response is not None:
        filters.append(pl.col("n_response") < min_response)
    if min_student is not None:
        filters.append(pl.col("n_student") < min_student)

    low_support_filter = filters[0]
    for filter_expr in filters[1:]:
        low_support_filter = low_support_filter | filter_expr

    return (
        df.group_by(ColumnMapping.estimate_question_id)
        .agg([
            pl.len().alias("n_response"),
            pl.col(ColumnMapping.student_id).n_unique().alias("n_student"),
        ])
        .filter(low_support_filter)
        .get_column(ColumnMapping.estimate_question_id)
        .to_list()
    )


def sanitize_training_data(
    df: pl.DataFrame,
    mastery_group_cols: list,
    min_obs: int,
    min_correct_rate: float = 0.1,
    max_correct_rate: float = 0.9,
    item_min_correct_rate: float | None = None,
    item_max_correct_rate: float | None = None,
    item_min_response: int | None = None,
    item_min_student: int | None = None,
    max_iterations: int = 20,
) -> tuple[pl.DataFrame, list, list[dict]]:
    stats = []
    if item_min_correct_rate is None:
        item_min_correct_rate = min_correct_rate
    if item_max_correct_rate is None:
        item_max_correct_rate = max_correct_rate

    for iteration in range(max_iterations):
        iteration_start_rows = len(df)

        steps = [
            (
                "mastery_min_obs",
                lambda data: remove_groups_with_insufficient_data(
                    data, mastery_group_cols, min_obs
                ),
            ),
            (
                "mastery_correct_rate",
                lambda data: remove_groups_outside_correct_rate_range(
                    data, mastery_group_cols, min_correct_rate, max_correct_rate
                ),
            ),
        ]

        for step_name, filter_step in steps:
            rows_before = len(df)
            df = filter_step(df)
            rows_after = len(df)
            stats.append({
                "iteration": iteration,
                "step": step_name,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "rows_removed": rows_before - rows_after,
            })

        if len(df) == iteration_start_rows:
            correct_rate_frozen_item_ids = set(get_frozen_item_ids_by_correct_rate(
                df,
                item_min_correct_rate,
                item_max_correct_rate,
            ))
            low_support_frozen_item_ids = set(get_frozen_item_ids_by_support(
                df,
                min_response=item_min_response,
                min_student=item_min_student,
            ))
            frozen_item_ids = sorted(correct_rate_frozen_item_ids | low_support_frozen_item_ids)
            stats.append({
                "iteration": iteration,
                "step": "item_parameter_freeze",
                "rows_before": len(df),
                "rows_after": len(df),
                "rows_removed": 0,
                "items_marked": len(frozen_item_ids),
                "items_marked_by_correct_rate": len(correct_rate_frozen_item_ids),
                "items_marked_by_low_support": len(low_support_frozen_item_ids),
            })
            return df, frozen_item_ids, stats

    raise RuntimeError(
        f"training data sanitization did not converge after {max_iterations} iterations"
    )


def batch_item_estimation(
    data: pl.DataFrame,
    default_values=None,
    tune_discrimination: bool = False,
    frozen_item_ids=None,
    **kwargs,
):
    if default_values is None:
        default_values = [0.0, 1.0]
    frozen_item_ids = set(frozen_item_ids or [])
    kwargs = dict(kwargs)

    # Per-iteration step-size trust regions retained for stability.
    # Absolute *_limit ceilings dropped; only hard floor is discrimination > 0.
    discrimination_lower = 1e-6
    difficulty_step_size = kwargs.pop("difficulty_step_size", 0.5)
    if tune_discrimination:
        discrimination_step_size = kwargs.pop("discrimination_step_size", 0.2)
    else:
        discrimination_step_size = kwargs.pop("discrimination_step_size", 0.01)
    if discrimination_step_size < 0:
        raise ValueError(
            f"discrimination_step_size must be non-negative, got {discrimination_step_size}"
        )

    def func(df: pl.DataFrame) -> pl.DataFrame:
        item_id = df[ColumnMapping.estimate_question_id][0]
        d0 = float(df[ColumnMapping.difficulty].mean())
        v0 = float(df[ColumnMapping.discrimination].mean())
        if item_id in frozen_item_ids:
            return pl.DataFrame({
                ColumnMapping.estimate_question_id: [item_id],
                "success": [1.0],
                ColumnMapping.difficulty: [d0],
                ColumnMapping.discrimination: [v0],
                ITEM_PARAMETER_FROZEN_COL: [True],
            })
        if v0 < discrimination_lower:
            v0 = discrimination_lower * 10
        m = df[ColumnMapping.mastery].to_numpy().astype(np.float64)
        r = df[ColumnMapping.score].to_numpy().astype(np.float64)
        call_kwargs = dict(kwargs)
        call_kwargs.update({
            "x0": [d0, v0],
            "method": "L-BFGS-B",
            "bounds": optimize.Bounds(
                [
                    d0 - difficulty_step_size,
                    max(v0 - discrimination_step_size, discrimination_lower),
                ],
                [
                    d0 + difficulty_step_size,
                    v0 + discrimination_step_size,
                ],
                keep_feasible=True,
            ),
        })
        opt_results = estimate_item(m, r, **call_kwargs)
        return pl.DataFrame({
            ColumnMapping.estimate_question_id: [item_id],
            "success": [float(opt_results.success)],
            ColumnMapping.difficulty: [float(opt_results.x[0])],
            ColumnMapping.discrimination: [float(opt_results.x[1])],
            ITEM_PARAMETER_FROZEN_COL: [False],
        })

    df_res = data.group_by(ColumnMapping.estimate_question_id).map_groups(func)

    df_res = df_res.with_columns([
        pl.when(pl.col("success") < 0.5)
            .then(pl.lit(default_values[0]))
            .otherwise(pl.col(ColumnMapping.difficulty))
            .alias(ColumnMapping.difficulty),
        pl.when(pl.col("success") < 0.5)
            .then(pl.lit(default_values[1]))
            .otherwise(pl.col(ColumnMapping.discrimination))
            .alias(ColumnMapping.discrimination),
    ])

    cols = [
        col for col in data.columns
        if col not in [
            "success",
            ColumnMapping.difficulty,
            ColumnMapping.discrimination,
            ITEM_PARAMETER_FROZEN_COL,
        ]
    ]

    return data.select(cols).join(df_res, on=ColumnMapping.estimate_question_id, how="inner")


def batch_mastery_estimation(
    data: pl.DataFrame,
    granularity_col=ColumnMapping.grade_strand_id,
    using_window_col: bool = False,
    default_value: float = 0.0,
    **kwargs,
):
    # Per-iteration step-size trust region retained for stability.
    # Absolute mastery_limit ceiling dropped; trust region floats with m0.
    mastery_step_size = kwargs.get("mastery_step_size", 1.0)

    group_cols = [ColumnMapping.student_id, granularity_col]
    if using_window_col:
        group_cols.append(ColumnMapping.window_index)

    def func(df: pl.DataFrame) -> pl.DataFrame:
        d = df[ColumnMapping.difficulty].to_numpy().astype(np.float64)
        v = df[ColumnMapping.discrimination].to_numpy().astype(np.float64)
        r = df[ColumnMapping.score].to_numpy().astype(np.float64)
        m0 = float(df[ColumnMapping.mastery].mean())
        call_kwargs = dict(kwargs)
        call_kwargs.update({
            "method": "bounded",
            "bounds": (m0 - mastery_step_size, m0 + mastery_step_size),
        })
        opt_results = estimate_mastery(d, v, r, **call_kwargs)
        row = {col: [df[col][0]] for col in group_cols}
        row["success"] = [float(opt_results.success)]
        row[ColumnMapping.mastery] = [float(opt_results.x)]
        return pl.DataFrame(row)

    df_res = data.group_by(group_cols).map_groups(func)

    df_res = df_res.with_columns(
        pl.when(pl.col("success") < 0.5)
            .then(pl.lit(default_value))
            .otherwise(pl.col(ColumnMapping.mastery))
            .alias(ColumnMapping.mastery)
    )

    cols = [col for col in data.columns if col not in ["success", ColumnMapping.mastery]]

    return data.select(cols).join(df_res, on=group_cols, how="inner")


def total_likelihood(df: pl.DataFrame) -> float:
    m = df[ColumnMapping.mastery].to_numpy().astype(np.float64)
    d = df[ColumnMapping.difficulty].to_numpy().astype(np.float64)
    v = df[ColumnMapping.discrimination].to_numpy().astype(np.float64)
    r = df[ColumnMapping.score].to_numpy().astype(np.float64)
    return float(likelihood(m, d, v, r))


def likelihood_denominator(df: pl.DataFrame) -> float:
    return float(len(df))
