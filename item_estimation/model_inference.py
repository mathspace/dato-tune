import logging
import math

import numpy as np
import pandas as pd
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


def split_train_test_data_on_group(df, group_cols, ratio=0.3):
    group_value_counts = df[group_cols].value_counts()
    idx_test = np.random.rand(group_value_counts.shape[0]) <= ratio
    idx_train = np.logical_not(idx_test)
    train_df = df.merge(
        group_value_counts.loc[idx_train, :].reset_index()[group_cols],
        on=group_cols,
        validate="m:1",
    )
    test_df = df.merge(
        group_value_counts.loc[idx_test, :].reset_index()[group_cols],
        on=group_cols,
        validate="m:1",
    )
    return train_df, test_df


def remove_groups_with_insufficient_data(df, group_cols, min_obs):
    group_value_counts = df[group_cols].value_counts()
    group_value_counts = group_value_counts[group_value_counts >= min_obs]
    return df.merge(
        group_value_counts.reset_index()[group_cols],
        on=group_cols,
        how="inner",
        validate="m:1",
    )


def batch_item_estimation(data, default_values=None, tune_discrimination=False, **kwargs):
    if default_values is None:
        default_values = [0.0, 1.0]

    # set bounds
    difficulty_step_size = kwargs.get("difficulty_step_size", 1.0)
    difficulty_limit = kwargs.get("difficulty_limit", (-5.0, 5.0))
    if tune_discrimination:
        discrimination_step_size = kwargs.get("discrimination_step_size", 0.1)
        discrimination_limit = kwargs.get("discrimination_limit", (0.0, 2.0))
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

    def func(df):
        m = df[ColumnMapping.mastery].values
        r = df[ColumnMapping.score].values
        d0 = df[ColumnMapping.difficulty].mean()
        v0 = df[ColumnMapping.discrimination].mean()
        kwargs["x0"] = [d0, v0]
        kwargs["method"] = "L-BFGS-B"
        kwargs["bounds"] = [get_difficulty_bounds(d0), get_discrimination_bounds(v0)]
        opt_results = estimate_item(m, r, **kwargs)
        out = np.zeros(len(opt_results.x) + 1)
        out[0] = opt_results.success
        out[1:] = opt_results.x
        return out

    res = data.groupby([ColumnMapping.question_id]).apply(func)
    df_res = pd.DataFrame(
        res.values.tolist(),
        columns=["success", ColumnMapping.difficulty, ColumnMapping.discrimination],
        index=res.index,
    )
    mask_failed = df_res.success < 0.5
    df_res.loc[mask_failed, ColumnMapping.difficulty] = default_values[0]
    df_res.loc[mask_failed, ColumnMapping.discrimination] = default_values[1]

    # Reset extreme discrimination values to prevent feedback loop
    mask_extreme = df_res[ColumnMapping.discrimination] >= 1.95
    n_reset = mask_extreme.sum()
    if n_reset > 0:
        logging.info(f"Resetting {n_reset} items with extreme discrimination values (>= 1.95) to 1.0")
    df_res.loc[mask_extreme, ColumnMapping.discrimination] = 1.0

    cols = [
        col
        for col in data.columns
        if col
        not in ["success", ColumnMapping.difficulty, ColumnMapping.discrimination]
    ]

    return pd.merge(
        data[cols], df_res.reset_index(), on=ColumnMapping.question_id, validate="m:1"
    )


def batch_mastery_estimation(
    data, granularity_col=ColumnMapping.grade_strand_id, using_window_col=False, default_value=0.0, **kwargs
):
    mastery_step_size = kwargs.get("mastery_step_size", 1.0)
    mastery_limit = kwargs.get("mastery_limit", (-5.0, 5.0))

    # Determine grouping columns
    group_cols = [ColumnMapping.student_id, granularity_col]
    if using_window_col is not None:
        group_cols.append(ColumnMapping.window_index)

    def set_bounds(m0):
        return (
            max(m0 - mastery_step_size, mastery_limit[0]),
            min(m0 + mastery_step_size, mastery_limit[1]),
        )

    def func(df):
        d = df[ColumnMapping.difficulty].values
        v = df[ColumnMapping.discrimination].values
        r = df[ColumnMapping.score].values
        m0 = df[ColumnMapping.mastery].mean()
        kwargs["method"] = "bounded"
        kwargs["bounds"] = set_bounds(m0)

        opt_results = estimate_mastery(d, v, r, **kwargs)
        out = np.zeros(2)
        out[0] = opt_results.success
        out[1] = opt_results.x
        return out

    res = data.groupby(group_cols).apply(func)
    df_res = pd.DataFrame(
        res.values.tolist(), columns=["success", ColumnMapping.mastery], index=res.index
    )
    mask_failed = df_res.success < 0.5
    df_res.loc[mask_failed, ColumnMapping.mastery] = default_value

    # Drop only mastery values that hit the exact bounds (optimizer constraint failure)
    mask_extreme = df_res[ColumnMapping.mastery].abs() >= 4.99
    n_dropped = mask_extreme.sum()
    if n_dropped > 0:
        logging.info(f"Dropping {n_dropped} student-mastery pairs that hit bounds (|mastery| >= 5.0)")
    df_res = df_res[~mask_extreme]

    cols = [
        col for col in data.columns if col not in ["success", ColumnMapping.mastery]
    ]

    return pd.merge(
        data[cols],
        df_res.reset_index(),
        on=group_cols,
        validate="m:1",
    )


def total_likelihood(df):
    m = df[ColumnMapping.mastery].values
    d = df[ColumnMapping.difficulty].values
    v = df[ColumnMapping.discrimination].values
    r = df[ColumnMapping.score].values
    return likelihood(m, d, v, r)
