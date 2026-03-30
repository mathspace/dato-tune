import logging
import math

import numpy as np
import polars as pl
from numba import float64, njit, vectorize
from scipy import optimize
from scipy.stats import logistic

from utils import ColumnMapping


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


@njit([float64(float64[:], float64, float64[:], float64[:])])
def likelihood_difficulty(m, difficulty, v, r):
    d = np.repeat(difficulty, len(r))
    return likelihood(m, d, v, r)


@njit([float64(float64[:], float64[:], float64, float64[:])])
def likelihood_discrimination(m, d, discrimination, r):
    v = np.repeat(discrimination, len(r))
    return likelihood(m, d, v, r)


@njit([float64(float64[:], float64, float64, float64[:])])
def likelihood_item(m, difficulty, discrimination, r):
    d = np.repeat(difficulty, len(r))
    v = np.repeat(discrimination, len(r))
    return likelihood(m, d, v, r)


## Numba estimation function
def estimate_mastery(d, v, r, **kwargs):
    def h(m):
        return -likelihood_mastery(m, d, v, r)

    return optimize.minimize_scalar(h, **kwargs)


def estimate_difficulty(m, v, r, **kwargs):
    def h(d):
        return -likelihood_difficulty(m, d, v, r)

    return optimize.minimize_scalar(h, **kwargs)


def estimate_item(m, r, **kwargs):
    def h(item_params):
        d, v = item_params
        return -likelihood_item(m, d, v, r)

    if "x0" not in kwargs.keys():
        kwargs["x0"] = [0.0, 1.0]
    return optimize.minimize(h, **kwargs)


## Scipy estimation function only for benchmark purpose


def likelihood_scipy(m, d, v, r):
    return np.sum(logistic.logcdf(v * (m - d)) * r) + np.sum(
        logistic.logsf(v * (m - d)) * (1 - r)
    )


def estimate_mastery_scipy(d, v, r, **kwargs):
    def h(m):
        return -likelihood_scipy(m, d, v, r)

    return optimize.minimize_scalar(h, **kwargs)


def estimate_difficulty_scipy(m, v, r, **kwargs):
    def h(d):
        return -likelihood_scipy(m, d, v, r)

    return optimize.minimize_scalar(h, **kwargs)


def estimate_item_scipy(m, r, **kwargs):
    def h(item_params):
        d, v = item_params
        return -likelihood_scipy(m, d, v, r)

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


def remove_groups_with_all_incorrect(df: pl.DataFrame, group_cols: list):
    has_correct = (
        df.group_by(group_cols)
        .agg((pl.col(ColumnMapping.score) > 0).any().alias("any_correct"))
        .filter(pl.col("any_correct"))
        .select(group_cols)
    )
    return df.join(has_correct, on=group_cols, how="inner")


def remove_groups_with_all_correct(df: pl.DataFrame, group_cols: list):
    has_incorrect = (
        df.group_by(group_cols)
        .agg((pl.col(ColumnMapping.score) == 0).any().alias("any_incorrect"))
        .filter(pl.col("any_incorrect"))
        .select(group_cols)
    )
    return df.join(has_incorrect, on=group_cols, how="inner")


def reset_extreme_discrimination(data: pl.DataFrame, threshold: float = 1.94, apply: bool = True) -> tuple[pl.DataFrame, bool]:
    total_items = data[ColumnMapping.estimate_question_id].n_unique()
    n_extreme = int(
        data.filter(pl.col(ColumnMapping.discrimination) >= threshold)
        [ColumnMapping.estimate_question_id].n_unique()
    )
    if n_extreme == 0:
        return data, False
    extreme_pct = 100 * n_extreme / total_items
    if extreme_pct <= 5.0:
        logging.info(
            f"Skipping discrimination reset: {n_extreme}/{total_items} items at cap ({extreme_pct:.2f}%) — within 5% threshold"
        )
        return data, False
    if not apply:
        return data, True
    logging.info(
        f"Resetting {n_extreme}/{total_items} items with extreme discrimination (>= {threshold}) to 1.0 ({extreme_pct:.2f}%)"
    )
    return data.with_columns(
        pl.when(pl.col(ColumnMapping.discrimination) >= threshold)
            .then(1.0)
            .otherwise(pl.col(ColumnMapping.discrimination))
            .alias(ColumnMapping.discrimination)
    ), True


def drop_extreme_mastery(data: pl.DataFrame, group_cols: list, threshold: float = 4.99, apply: bool = True) -> tuple[pl.DataFrame, bool]:
    total_groups = data.select(group_cols).unique().height
    n_dropped = (
        data.filter(pl.col(ColumnMapping.mastery).abs() >= threshold)
        .select(group_cols)
        .unique()
        .height
    )
    if n_dropped == 0:
        return data, False
    drop_pct = 100 * n_dropped / total_groups
    if drop_pct <= 0.5:
        return data, False
    if not apply:
        return data, True
    logging.info(
        f"Dropping {n_dropped} unique student groups that hit bounds (|mastery| >= {threshold}) "
        f"({drop_pct:.2f}% of {total_groups} groups)"
    )
    return data.filter(pl.col(ColumnMapping.mastery).abs() < threshold), True


def batch_item_estimation(
    data: pl.DataFrame, default_values=None, tune_discrimination: bool = False, **kwargs
):
    if default_values is None:
        default_values = [0.0, 1.0]

    difficulty_step_size = kwargs.get("difficulty_step_size", 0.5)
    difficulty_limit = kwargs.get("difficulty_limit", (-3.0, 3.0))
    if tune_discrimination:
        discrimination_step_size = kwargs.get("discrimination_step_size", 0.1)
        discrimination_limit = kwargs.get("discrimination_limit", (0.0, 1.95))
    else:
        discrimination_step_size = kwargs.get("discrimination_step_size", 0.01)
        discrimination_limit = kwargs.get("discrimination_limit", (0.95, 1.05))

    def get_difficulty_bounds(d0):
        return (
            max(d0 - difficulty_step_size, difficulty_limit[0]),
            min(d0 + difficulty_step_size, difficulty_limit[1]),
        )

    def get_discrimination_bounds(v0):
        return (
            max(v0 * (1 - discrimination_step_size), discrimination_limit[0]),
            min(v0 * (1 + discrimination_step_size), discrimination_limit[1]),
        )

    def func(df: pl.DataFrame) -> pl.DataFrame:
        m = df[ColumnMapping.mastery].to_numpy().astype(np.float64)
        r = df[ColumnMapping.score].to_numpy().astype(np.float64)
        d0 = float(df[ColumnMapping.difficulty].mean())
        v0 = float(df[ColumnMapping.discrimination].mean())
        call_kwargs = dict(kwargs)
        difficulty_bounds = get_difficulty_bounds(d0)
        discrimination_bounds = get_discrimination_bounds(v0)
        call_kwargs.update({
            "x0": [d0, v0],
            "method": "L-BFGS-B",
            "bounds": optimize.Bounds(
                [difficulty_bounds[0], discrimination_bounds[0]],
                [difficulty_bounds[1], discrimination_bounds[1]],
                keep_feasible=True,
            ),
        })
        opt_results = estimate_item(m, r, **call_kwargs)
        return pl.DataFrame({
            ColumnMapping.estimate_question_id: [df[ColumnMapping.estimate_question_id][0]],
            "success": [float(opt_results.success)],
            ColumnMapping.difficulty: [float(opt_results.x[0])],
            ColumnMapping.discrimination: [float(opt_results.x[1])],
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
        if col not in ["success", ColumnMapping.difficulty, ColumnMapping.discrimination]
    ]

    return data.select(cols).join(df_res, on=ColumnMapping.estimate_question_id, how="inner")


def batch_mastery_estimation(
    data: pl.DataFrame,
    granularity_col=ColumnMapping.grade_strand_id,
    using_window_col: bool = False,
    default_value: float = 0.0,
    **kwargs,
):
    mastery_step_size = kwargs.get("mastery_step_size", 1.0)
    mastery_limit = kwargs.get("mastery_limit", (-5.0, 5.0))

    group_cols = [ColumnMapping.student_id, granularity_col]
    if using_window_col:
        group_cols.append(ColumnMapping.window_index)

    def set_bounds(m0):
        return (
            max(m0 - mastery_step_size, mastery_limit[0]),
            min(m0 + mastery_step_size, mastery_limit[1]),
        )

    def func(df: pl.DataFrame) -> pl.DataFrame:
        d = df[ColumnMapping.difficulty].to_numpy().astype(np.float64)
        v = df[ColumnMapping.discrimination].to_numpy().astype(np.float64)
        r = df[ColumnMapping.score].to_numpy().astype(np.float64)
        m0 = float(df[ColumnMapping.mastery].mean())
        call_kwargs = dict(kwargs)
        call_kwargs.update({
            "method": "bounded",
            "bounds": set_bounds(m0),
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
