import math
from collections.abc import Callable
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

DEFAULT_C = 0.5
DEFAULT_L = 0.0

_STATS_COUNT = 0
_STATS_SOURCE_SUM = 1
_STATS_SOURCE_SQUARE_SUM = 2
_STATS_WEIGHT_SUM = 3
_STATS_WEIGHTED_SOURCE_SUM = 4
_STATS_WEIGHTED_TARGET_SUM = 5
_STATS_WEIGHTED_SOURCE_SQUARE_SUM = 6
_STATS_WEIGHTED_SOURCE_TARGET_SUM = 7
_STATS_INVALID = 8
_WLS_STATS_SIZE = 9
_ProgressCallback = Callable[[int, int], None]


def optimise_wls_coefficients(
    node_pair_values: NDArray[np.float64] | None,
    reverse_pair: bool = False,
    min_shared_students: int = 2,
    min_source_variance: float = 1e-8,
) -> dict[str, float]:
    if node_pair_values is None:
        return {"L": DEFAULT_L, "C": DEFAULT_C}
    if len(node_pair_values) < min_shared_students:
        return {"L": DEFAULT_L, "C": DEFAULT_C}

    if reverse_pair:
        mu_j, mu_i, sig_j, sig_i = (
            node_pair_values[:, 0],
            node_pair_values[:, 1],
            node_pair_values[:, 2],
            node_pair_values[:, 3],
        )
    else:
        mu_i, mu_j, sig_i, sig_j = (
            node_pair_values[:, 0],
            node_pair_values[:, 1],
            node_pair_values[:, 2],
            node_pair_values[:, 3],
        )

    if not np.isfinite(node_pair_values).all():
        return {"L": DEFAULT_L, "C": DEFAULT_C}

    uncertainty = (sig_j * sig_j) + (sig_i * sig_i)
    if not np.isfinite(uncertainty).all() or np.any(uncertainty <= 0.0):
        return {"L": DEFAULT_L, "C": DEFAULT_C}
    if np.var(mu_j) < min_source_variance:
        return {"L": DEFAULT_L, "C": DEFAULT_C}

    w = 1.0 / uncertainty
    weight_sum = w.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return {"L": DEFAULT_L, "C": DEFAULT_C}

    mu_j_mean = np.sum(w * mu_j) / weight_sum
    mu_i_mean = np.sum(w * mu_i) / weight_sum
    centered_mu_j = mu_j - mu_j_mean
    denominator = np.sum(w * centered_mu_j * centered_mu_j)
    if not np.isfinite(denominator) or denominator < min_source_variance:
        return {"L": DEFAULT_L, "C": DEFAULT_C}

    C = np.sum(w * centered_mu_j * (mu_i - mu_i_mean)) / denominator
    L = mu_i_mean - C * mu_j_mean
    if not np.isfinite(L) or not np.isfinite(C):
        return {"L": DEFAULT_L, "C": DEFAULT_C}

    return {"L": float(L), "C": float(C)}


def _optimise_wls_coefficients_from_stats(
    stats: list[float] | None,
    min_shared_students: int = 2,
    min_source_variance: float = 1e-8,
) -> dict[str, float]:
    if stats is None:
        return {"L": DEFAULT_L, "C": DEFAULT_C}
    if stats[_STATS_INVALID] > 0.0:
        return {"L": DEFAULT_L, "C": DEFAULT_C}

    count = stats[_STATS_COUNT]
    if count < min_shared_students:
        return {"L": DEFAULT_L, "C": DEFAULT_C}

    source_mean = stats[_STATS_SOURCE_SUM] / count
    source_variance = (
        stats[_STATS_SOURCE_SQUARE_SUM] / count - source_mean * source_mean
    )
    if not math.isfinite(source_variance) or source_variance < min_source_variance:
        return {"L": DEFAULT_L, "C": DEFAULT_C}

    weight_sum = stats[_STATS_WEIGHT_SUM]
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        return {"L": DEFAULT_L, "C": DEFAULT_C}

    weighted_source_mean = stats[_STATS_WEIGHTED_SOURCE_SUM] / weight_sum
    weighted_target_mean = stats[_STATS_WEIGHTED_TARGET_SUM] / weight_sum
    denominator = (
        stats[_STATS_WEIGHTED_SOURCE_SQUARE_SUM]
        - stats[_STATS_WEIGHTED_SOURCE_SUM]
        * stats[_STATS_WEIGHTED_SOURCE_SUM]
        / weight_sum
    )
    if not math.isfinite(denominator) or denominator < min_source_variance:
        return {"L": DEFAULT_L, "C": DEFAULT_C}

    numerator = (
        stats[_STATS_WEIGHTED_SOURCE_TARGET_SUM]
        - stats[_STATS_WEIGHTED_SOURCE_SUM]
        * stats[_STATS_WEIGHTED_TARGET_SUM]
        / weight_sum
    )
    C = numerator / denominator
    L = weighted_target_mean - C * weighted_source_mean
    if not math.isfinite(L) or not math.isfinite(C):
        return {"L": DEFAULT_L, "C": DEFAULT_C}

    return {"L": float(L), "C": float(C)}


def get_wls_parameter_rows(
    student_estimation_models: dict[str, dict[str, dict[str, float]]],
    *,
    min_shared_students: int = 2,
    min_source_variance: float = 1e-8,
    stats_progress_callback: _ProgressCallback | None = None,
    pair_progress_callback: _ProgressCallback | None = None,
) -> list[tuple[str, str, float, float]]:
    sorted_nodes = sorted(
        {node for node_models in student_estimation_models.values() for node in node_models}
    )
    node_to_index = {node: index for index, node in enumerate(sorted_nodes)}
    stats = np.zeros(
        (_WLS_STATS_SIZE, len(sorted_nodes), len(sorted_nodes)),
        dtype=np.float64,
    )
    total_students = len(student_estimation_models)
    for students_processed, node_models in enumerate(
        student_estimation_models.values(), start=1
    ):
        _accumulate_student_wls_stats(stats, node_to_index, node_models)

        if stats_progress_callback is not None:
            stats_progress_callback(students_processed, total_students)

    parameter_rows: list[tuple[str, str, float, float]] = []
    total_pair_fits = len(sorted_nodes) * (len(sorted_nodes) - 1) // 2
    for pair_fits_processed, (source_node, target_node) in enumerate(
        combinations(sorted_nodes, 2), start=1
    ):
        source_index = node_to_index[source_node]
        target_index = node_to_index[target_node]
        coeffs = _optimise_wls_coefficients_from_stats(
            stats[:, source_index, target_index],
            min_shared_students=min_shared_students,
            min_source_variance=min_source_variance,
        )
        reverse_coeffs = _optimise_wls_coefficients_from_stats(
            stats[:, target_index, source_index],
            min_shared_students=min_shared_students,
            min_source_variance=min_source_variance,
        )
        parameter_rows.append(
            (source_node, target_node, float(coeffs["L"]), float(coeffs["C"]))
        )
        parameter_rows.append(
            (
                target_node,
                source_node,
                float(reverse_coeffs["L"]),
                float(reverse_coeffs["C"]),
            )
        )
        if pair_progress_callback is not None:
            pair_progress_callback(pair_fits_processed, total_pair_fits)

    return parameter_rows


def _accumulate_student_wls_stats(
    stats: NDArray[np.float64],
    node_to_index: dict[str, int],
    node_models: dict[str, dict[str, float]],
):
    if len(node_models) < 2:
        return

    indexes = np.fromiter(
        (node_to_index[node] for node in node_models),
        dtype=np.int64,
        count=len(node_models),
    )
    means = np.fromiter(
        (model["mean"] for model in node_models.values()),
        dtype=np.float64,
        count=len(node_models),
    )
    stds = np.fromiter(
        (model["std"] for model in node_models.values()),
        dtype=np.float64,
        count=len(node_models),
    )

    index_grid = np.ix_(indexes, indexes)
    source_means = means[:, None]
    target_means = means[None, :]
    source_stds = stds[:, None]
    target_stds = stds[None, :]
    uncertainty = source_stds * source_stds + target_stds * target_stds
    valid = (
        np.isfinite(source_means)
        & np.isfinite(target_means)
        & np.isfinite(source_stds)
        & np.isfinite(target_stds)
        & np.isfinite(uncertainty)
        & (uncertainty > 0.0)
    )

    weights = np.zeros_like(uncertainty)
    np.divide(1.0, uncertainty, out=weights, where=valid)

    stats[_STATS_COUNT][index_grid] += 1.0
    stats[_STATS_INVALID][index_grid] += ~valid
    stats[_STATS_SOURCE_SUM][index_grid] += source_means
    stats[_STATS_SOURCE_SQUARE_SUM][index_grid] += source_means * source_means
    stats[_STATS_WEIGHT_SUM][index_grid] += weights
    stats[_STATS_WEIGHTED_SOURCE_SUM][index_grid] += weights * source_means
    stats[_STATS_WEIGHTED_TARGET_SUM][index_grid] += weights * target_means
    stats[_STATS_WEIGHTED_SOURCE_SQUARE_SUM][index_grid] += (
        weights * source_means * source_means
    )
    stats[_STATS_WEIGHTED_SOURCE_TARGET_SUM][index_grid] += (
        weights * source_means * target_means
    )
