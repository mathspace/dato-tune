"""
previously implemented Bayes IRT functions, copied from repo sana-data-science-2
modified to streamline and improve performance
"""

import numpy as np
from numba import boolean, float64, njit
from scipy.special import ndtr


MIN_PROBA_SPACE = 0.0001
MAX_PROBA_SPACE = 0.9999

MAX_LOGIT_SIGMA = 5.0
MIN_LOGIT_SIGMA = 0.1
N_RESOLUTION = 10000

MAX_S = 10.0


@njit([float64(float64)], cache=True)
def logit(x):
    return np.log(x) - np.log(1.0 - x)


@njit([float64(float64)], cache=True)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


MIN_LOGIT_SPACE = logit(MIN_PROBA_SPACE)
MAX_LOGIT_SPACE = logit(MAX_PROBA_SPACE)


# numba doesn't like constants defined outside of njit functions,
# it makes caching ineffecient as it has to rederive values and functions each time.
def make_fast_normal(min_logit, max_logit, n_resolution, min_proba, max_proba):
    """Factory function that creates a fast_normal with baked-in grids."""
    real_grid = np.linspace(min_logit, max_logit, n_resolution + 1)
    normal_grid = ndtr(real_grid)
    step_size = (max_logit - min_logit) / n_resolution

    @njit(float64(float64), cache=True)
    def fast_normal(x):
        minGrid = real_grid[0]
        maxGrid = real_grid[-1]

        if x <= minGrid:
            return min_proba
        elif x >= maxGrid:
            return max_proba
        else:
            # linear interpolation between grid points
            idx = int((x - minGrid) / step_size)
            # closer to 1 as distance to left grid point increases
            right_weight = (x - real_grid[idx]) / step_size

            return (
                normal_grid[idx] * (1.0 - right_weight)
                + normal_grid[idx + 1] * right_weight
            )

    return fast_normal


# Create the cached version once at module level
fast_normal = make_fast_normal(
    MIN_LOGIT_SPACE, MAX_LOGIT_SPACE, N_RESOLUTION, MIN_PROBA_SPACE, MAX_PROBA_SPACE
)


@njit([float64(float64)], cache=True)
def safe_sqrt(x):
    if x <= 0.0:
        return 10 ** (-6)
    else:
        return np.sqrt(x)


@njit(float64(float64, float64, float64), cache=True)
def numba_clip(x, lower, upper):
    if x <= lower:
        return lower
    elif x >= upper:
        return upper
    return x


def make_numba_clip(lower, upper):
    """Factory function that creates a specialized clip function with baked-in bounds."""

    @njit(float64(float64), cache=True)
    def clip(x):
        if x <= lower:
            return lower
        elif x >= upper:
            return upper
        return x

    return clip


# Create specialized clip functions once at module level
numba_clip_proba = make_numba_clip(MIN_PROBA_SPACE, MAX_PROBA_SPACE)
numba_clip_logit = make_numba_clip(MIN_LOGIT_SPACE, MAX_LOGIT_SPACE)
numba_clip_logit_sigma = make_numba_clip(MIN_LOGIT_SIGMA, MAX_LOGIT_SIGMA)
numba_clip_s = make_numba_clip(0.0, MAX_S)


@njit(
    float64[:](
        float64,
        float64,
        float64,
        float64,
        boolean,
    ),
    cache=True,
)
def numba_update(
    mu_infer: float,
    sigma_infer: float,
    mu_observe: float,
    sigma_observe: float,
    correct: bool,
):
    # See section 8.1.2, eq 38-45 for details on the following quantities.
    s = numba_clip_s(sigma_infer / sigma_observe)
    r_max = np.maximum(s, 1.0) * 4.0
    r = numba_clip((mu_infer - mu_observe) / sigma_observe, -r_max, r_max)
    factor_in_exp = r**2.0 / (2.0 * (1.0 + s**2.0))
    z_normalizer = numba_clip_proba(fast_normal(r / np.sqrt(1.0 + s**2.0)))
    n_value_in_exp = s / np.sqrt(2.0 * np.pi) * np.exp(-factor_in_exp)
    sigma_square = sigma_infer**2
    sigma_post_square = sigma_observe**2
    sigma_square_sum = sigma_square + sigma_post_square
    sigma_square_pooled = sigma_square * sigma_post_square / sigma_square_sum
    sigma_pooled = np.sqrt(sigma_square_pooled)
    mu_pooled = (
        mu_infer * sigma_post_square + mu_observe * sigma_square
    ) / sigma_square_sum

    if correct:
        n_over_z = n_value_in_exp / z_normalizer
        mu_updated = numba_clip_logit(mu_infer + sigma_pooled * n_over_z)
        sigma_updated = numba_clip_logit_sigma(
            safe_sqrt(
                sigma_square
                + sigma_pooled * (mu_pooled - mu_infer) * n_over_z
                - sigma_square_pooled * n_over_z**2
            )
        )

    else:
        n_over_z = n_value_in_exp / (1.0 - z_normalizer)
        mu_updated = numba_clip_logit(mu_infer - sigma_pooled * n_over_z)
        sigma_updated = numba_clip_logit_sigma(
            safe_sqrt(
                sigma_square
                - sigma_pooled * (mu_pooled - mu_infer) * n_over_z
                - sigma_square_pooled * n_over_z**2
            )
        )

    return np.array([mu_updated, sigma_updated, numba_clip_proba(sigmoid(mu_updated))])


@njit(
    float64[:](
        float64,
        float64,
        float64,
        float64,
        boolean,
    ),
    cache=True,
)
def numba_event_update(
    ability: float,
    ability_sigma: float,
    difficulty: float,
    discriminative_index: float,
    response: bool,
):
    mu_infer = logit(numba_clip_proba(ability))
    mu_observe = logit(numba_clip_proba(difficulty))
    sigma_infer = numba_clip_logit_sigma(ability_sigma)
    sigma_observe = 0.47 * np.pi / discriminative_index
    result = numba_update(
        mu_infer,
        sigma_infer,
        mu_observe,
        sigma_observe,
        response,
    )
    return result
