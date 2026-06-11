import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

DEFAULT_C = 0.5
DEFAULT_L = 0.0
DEFAULT_MIN_SHARED_STUDENTS = 10
DEFAULT_MIN_SOURCE_VARIANCE = 0.1
DEFAULT_MAX_WLS_WEIGHT = 5.0
DEFAULT_UNSTABLE_SLOPE_THRESHOLD = 1.5
DEFAULT_UNSTABLE_SLOPE_MIN_SHARED_STUDENTS = 30
DEFAULT_UNSTABLE_SLOPE_MIN_DENOMINATOR = 5.0
MIN_PROPAGATION_COEFFICIENT = 0.0

_STATS_COUNT = 0
_STATS_SOURCE_SUM = 1
_STATS_SOURCE_SQUARE_SUM = 2
_STATS_WEIGHT_SUM = 3
_STATS_WEIGHTED_SOURCE_SUM = 4
_STATS_WEIGHTED_TARGET_SUM = 5
_STATS_WEIGHTED_SOURCE_SQUARE_SUM = 6
_STATS_WEIGHTED_SOURCE_TARGET_SUM = 7
_STATS_INVALID_INPUT_COUNT = 8
_STATS_INVALID_UNCERTAINTY_COUNT = 9
_WLS_STATS_SIZE = 10
_ProgressCallback = Callable[[int, int], None]
WLSParameterRow = tuple[
    str,
    str,
    float,
    float,
    str | None,
    int,
    int,
    int,
    float | None,
    float | None,
    float | None,
]


@dataclass(frozen=True)
class WLSFitResult:
    L: float
    C: float
    default_reason: str | None
    shared_students: int
    invalid_input_count: int
    invalid_uncertainty_count: int
    source_variance: float | None
    weight_sum: float | None
    denominator: float | None


@dataclass(frozen=True)
class WLSParameterResult:
    parameter_rows: list[WLSParameterRow]
    default_count: int
    fit_count: int


def optimise_wls_coefficients(
    node_pair_values: NDArray[np.float64] | None,
    reverse_pair: bool = False,
    min_shared_students: int = DEFAULT_MIN_SHARED_STUDENTS,
    min_source_variance: float = DEFAULT_MIN_SOURCE_VARIANCE,
    max_wls_weight: float = DEFAULT_MAX_WLS_WEIGHT,
    unstable_slope_threshold: float = DEFAULT_UNSTABLE_SLOPE_THRESHOLD,
    unstable_slope_min_shared_students: int = DEFAULT_UNSTABLE_SLOPE_MIN_SHARED_STUDENTS,
    unstable_slope_min_denominator: float = DEFAULT_UNSTABLE_SLOPE_MIN_DENOMINATOR,
) -> dict[str, float]:
    """
    Fit weighted propagation parameters for target ability from source ability.

    Negative slopes are sanitised to 0 so propagation does not invert ability
    updates. Positive slopes are left unconstrained so large fitted C values
    remain visible for analysis. L is recalculated after sanitising C, and is
    set to 0 when C is 0 because the propagation term is inactive.
    """
    if not math.isfinite(max_wls_weight) or max_wls_weight <= 0.0:
        raise ValueError("max_wls_weight must be positive")
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

    w = np.minimum(1.0 / uncertainty, max_wls_weight)
    weight_sum = w.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return {"L": DEFAULT_L, "C": DEFAULT_C}

    mu_j_mean = np.sum(w * mu_j) / weight_sum
    mu_i_mean = np.sum(w * mu_i) / weight_sum
    centered_mu_j = mu_j - mu_j_mean
    denominator = np.sum(w * centered_mu_j * centered_mu_j)
    if not np.isfinite(denominator) or denominator < min_source_variance:
        return {"L": DEFAULT_L, "C": DEFAULT_C}

    raw_C = np.sum(w * centered_mu_j * (mu_i - mu_i_mean)) / denominator
    C = max(raw_C, MIN_PROPAGATION_COEFFICIENT)
    L = DEFAULT_L if C == MIN_PROPAGATION_COEFFICIENT else mu_i_mean - C * mu_j_mean
    if not np.isfinite(L) or not np.isfinite(raw_C):
        return {"L": DEFAULT_L, "C": DEFAULT_C}
    if _is_unstable_large_slope(
        C,
        len(node_pair_values),
        denominator,
        unstable_slope_threshold,
        unstable_slope_min_shared_students,
        unstable_slope_min_denominator,
    ):
        return {"L": DEFAULT_L, "C": DEFAULT_C}

    return {"L": float(L), "C": float(C)}


def _optimise_wls_coefficients_from_stats(
    stats: list[float] | None,
    min_shared_students: int = DEFAULT_MIN_SHARED_STUDENTS,
    min_source_variance: float = DEFAULT_MIN_SOURCE_VARIANCE,
    unstable_slope_threshold: float = DEFAULT_UNSTABLE_SLOPE_THRESHOLD,
    unstable_slope_min_shared_students: int = DEFAULT_UNSTABLE_SLOPE_MIN_SHARED_STUDENTS,
    unstable_slope_min_denominator: float = DEFAULT_UNSTABLE_SLOPE_MIN_DENOMINATOR,
) -> WLSFitResult:
    if stats is None:
        return _default_wls_fit_result("missing_pair_stats")

    count = int(stats[_STATS_COUNT])
    invalid_input_count = int(stats[_STATS_INVALID_INPUT_COUNT])
    invalid_uncertainty_count = int(stats[_STATS_INVALID_UNCERTAINTY_COUNT])
    if invalid_input_count > 0:
        return _default_wls_fit_result(
            "non_finite_input",
            shared_students=count,
            invalid_input_count=invalid_input_count,
            invalid_uncertainty_count=invalid_uncertainty_count,
        )
    if invalid_uncertainty_count > 0:
        return _default_wls_fit_result(
            "non_positive_uncertainty",
            shared_students=count,
            invalid_input_count=invalid_input_count,
            invalid_uncertainty_count=invalid_uncertainty_count,
        )
    if count < min_shared_students:
        return _default_wls_fit_result(
            "insufficient_shared_students",
            shared_students=count,
            invalid_input_count=invalid_input_count,
            invalid_uncertainty_count=invalid_uncertainty_count,
        )

    source_mean = stats[_STATS_SOURCE_SUM] / count
    source_variance = (
        stats[_STATS_SOURCE_SQUARE_SUM] / count - source_mean * source_mean
    )
    if not math.isfinite(source_variance) or source_variance < min_source_variance:
        return _default_wls_fit_result(
            "insufficient_source_variance",
            shared_students=count,
            invalid_input_count=invalid_input_count,
            invalid_uncertainty_count=invalid_uncertainty_count,
            source_variance=source_variance,
        )

    weight_sum = stats[_STATS_WEIGHT_SUM]
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        return _default_wls_fit_result(
            "non_positive_weight_sum",
            shared_students=count,
            invalid_input_count=invalid_input_count,
            invalid_uncertainty_count=invalid_uncertainty_count,
            source_variance=source_variance,
            weight_sum=weight_sum,
        )

    weighted_source_mean = stats[_STATS_WEIGHTED_SOURCE_SUM] / weight_sum
    weighted_target_mean = stats[_STATS_WEIGHTED_TARGET_SUM] / weight_sum
    denominator = (
        stats[_STATS_WEIGHTED_SOURCE_SQUARE_SUM]
        - stats[_STATS_WEIGHTED_SOURCE_SUM]
        * stats[_STATS_WEIGHTED_SOURCE_SUM]
        / weight_sum
    )
    if not math.isfinite(denominator) or denominator < min_source_variance:
        return _default_wls_fit_result(
            "insufficient_denominator",
            shared_students=count,
            invalid_input_count=invalid_input_count,
            invalid_uncertainty_count=invalid_uncertainty_count,
            source_variance=source_variance,
            weight_sum=weight_sum,
            denominator=denominator,
        )

    numerator = (
        stats[_STATS_WEIGHTED_SOURCE_TARGET_SUM]
        - stats[_STATS_WEIGHTED_SOURCE_SUM]
        * stats[_STATS_WEIGHTED_TARGET_SUM]
        / weight_sum
    )
    raw_C = numerator / denominator
    C = max(raw_C, MIN_PROPAGATION_COEFFICIENT)
    L = (
        DEFAULT_L
        if C == MIN_PROPAGATION_COEFFICIENT
        else weighted_target_mean - C * weighted_source_mean
    )
    if not math.isfinite(L) or not math.isfinite(raw_C):
        return _default_wls_fit_result(
            "non_finite_coefficients",
            shared_students=count,
            invalid_input_count=invalid_input_count,
            invalid_uncertainty_count=invalid_uncertainty_count,
            source_variance=source_variance,
            weight_sum=weight_sum,
            denominator=denominator,
        )
    if _is_unstable_large_slope(
        C,
        count,
        denominator,
        unstable_slope_threshold,
        unstable_slope_min_shared_students,
        unstable_slope_min_denominator,
    ):
        return _default_wls_fit_result(
            "unstable_large_slope",
            shared_students=count,
            invalid_input_count=invalid_input_count,
            invalid_uncertainty_count=invalid_uncertainty_count,
            source_variance=source_variance,
            weight_sum=weight_sum,
            denominator=denominator,
        )

    return WLSFitResult(
        L=float(L),
        C=float(C),
        default_reason=None,
        shared_students=count,
        invalid_input_count=invalid_input_count,
        invalid_uncertainty_count=invalid_uncertainty_count,
        source_variance=float(source_variance),
        weight_sum=float(weight_sum),
        denominator=float(denominator),
    )


def _default_wls_fit_result(
    default_reason: str,
    *,
    shared_students: int = 0,
    invalid_input_count: int = 0,
    invalid_uncertainty_count: int = 0,
    source_variance: float | None = None,
    weight_sum: float | None = None,
    denominator: float | None = None,
) -> WLSFitResult:
    return WLSFitResult(
        L=DEFAULT_L,
        C=DEFAULT_C,
        default_reason=default_reason,
        shared_students=shared_students,
        invalid_input_count=invalid_input_count,
        invalid_uncertainty_count=invalid_uncertainty_count,
        source_variance=source_variance,
        weight_sum=weight_sum,
        denominator=denominator,
    )


def _is_unstable_large_slope(
    C: float,
    shared_students: int,
    denominator: float,
    unstable_slope_threshold: float,
    unstable_slope_min_shared_students: int,
    unstable_slope_min_denominator: float,
) -> bool:
    return C > unstable_slope_threshold and (
        shared_students < unstable_slope_min_shared_students
        or denominator < unstable_slope_min_denominator
    )


def get_wls_parameter_rows(
    student_estimation_models: dict[str, dict[str, dict[str, float]]],
    *,
    allowed_node_pairs: set[tuple[str, str]],
    min_shared_students: int = DEFAULT_MIN_SHARED_STUDENTS,
    min_source_variance: float = DEFAULT_MIN_SOURCE_VARIANCE,
    max_wls_weight: float = DEFAULT_MAX_WLS_WEIGHT,
    unstable_slope_threshold: float = DEFAULT_UNSTABLE_SLOPE_THRESHOLD,
    unstable_slope_min_shared_students: int = DEFAULT_UNSTABLE_SLOPE_MIN_SHARED_STUDENTS,
    unstable_slope_min_denominator: float = DEFAULT_UNSTABLE_SLOPE_MIN_DENOMINATOR,
    stats_progress_callback: _ProgressCallback | None = None,
    pair_progress_callback: _ProgressCallback | None = None,
) -> list[WLSParameterRow]:
    result = get_wls_parameter_result(
        student_estimation_models,
        allowed_node_pairs=allowed_node_pairs,
        min_shared_students=min_shared_students,
        min_source_variance=min_source_variance,
        max_wls_weight=max_wls_weight,
        unstable_slope_threshold=unstable_slope_threshold,
        unstable_slope_min_shared_students=unstable_slope_min_shared_students,
        unstable_slope_min_denominator=unstable_slope_min_denominator,
        stats_progress_callback=stats_progress_callback,
        pair_progress_callback=pair_progress_callback,
    )
    return result.parameter_rows


def get_wls_parameter_result(
    student_estimation_models: dict[str, dict[str, dict[str, float]]],
    *,
    allowed_node_pairs: set[tuple[str, str]],
    min_shared_students: int = DEFAULT_MIN_SHARED_STUDENTS,
    min_source_variance: float = DEFAULT_MIN_SOURCE_VARIANCE,
    max_wls_weight: float = DEFAULT_MAX_WLS_WEIGHT,
    unstable_slope_threshold: float = DEFAULT_UNSTABLE_SLOPE_THRESHOLD,
    unstable_slope_min_shared_students: int = DEFAULT_UNSTABLE_SLOPE_MIN_SHARED_STUDENTS,
    unstable_slope_min_denominator: float = DEFAULT_UNSTABLE_SLOPE_MIN_DENOMINATOR,
    stats_progress_callback: _ProgressCallback | None = None,
    pair_progress_callback: _ProgressCallback | None = None,
) -> WLSParameterResult:
    if not math.isfinite(max_wls_weight) or max_wls_weight <= 0.0:
        raise ValueError("max_wls_weight must be positive")

    sorted_nodes = sorted(
        {node for node_models in student_estimation_models.values() for node in node_models}
    )
    node_to_index = {node: index for index, node in enumerate(sorted_nodes)}
    allowed_pair_indexes = _get_allowed_pair_indexes(
        node_to_index,
        allowed_node_pairs,
    )
    allowed_target_indexes_by_source = _get_allowed_target_indexes_by_source(
        allowed_pair_indexes
    )
    stats = np.zeros(
        (_WLS_STATS_SIZE, len(sorted_nodes), len(sorted_nodes)),
        dtype=np.float64,
    )
    total_students = len(student_estimation_models)
    for students_processed, node_models in enumerate(
        student_estimation_models.values(), start=1
    ):
        _accumulate_student_wls_stats(
            stats,
            node_to_index,
            node_models,
            allowed_target_indexes_by_source,
            max_wls_weight,
        )

        if stats_progress_callback is not None:
            stats_progress_callback(students_processed, total_students)

    parameter_rows: list[WLSParameterRow] = []
    default_count = 0
    total_pair_fits = len(allowed_pair_indexes)
    for pair_fits_processed, (source_index, target_index) in enumerate(
        allowed_pair_indexes, start=1
    ):
        source_node = sorted_nodes[source_index]
        target_node = sorted_nodes[target_index]
        fit = _optimise_wls_coefficients_from_stats(
            stats[:, source_index, target_index],
            min_shared_students=min_shared_students,
            min_source_variance=min_source_variance,
            unstable_slope_threshold=unstable_slope_threshold,
            unstable_slope_min_shared_students=unstable_slope_min_shared_students,
            unstable_slope_min_denominator=unstable_slope_min_denominator,
        )
        if fit.default_reason is not None:
            default_count += 1
        parameter_rows.append(
            (
                source_node,
                target_node,
                fit.L,
                fit.C,
                fit.default_reason,
                fit.shared_students,
                fit.invalid_input_count,
                fit.invalid_uncertainty_count,
                fit.source_variance,
                fit.weight_sum,
                fit.denominator,
            )
        )
        if pair_progress_callback is not None:
            pair_progress_callback(pair_fits_processed, total_pair_fits)

    return WLSParameterResult(
        parameter_rows=parameter_rows,
        default_count=default_count,
        fit_count=len(parameter_rows) - default_count,
    )


def _accumulate_student_wls_stats(
    stats: NDArray[np.float64],
    node_to_index: dict[str, int],
    node_models: dict[str, dict[str, float]],
    allowed_target_indexes_by_source: dict[int, frozenset[int]],
    max_wls_weight: float,
):
    if len(node_models) < 2:
        return

    student_indexes_by_node = {
        node_to_index[node]: model for node, model in node_models.items()
    }
    student_indexes = set(student_indexes_by_node)
    for source_index, source_model in student_indexes_by_node.items():
        target_indexes = allowed_target_indexes_by_source.get(source_index)
        if not target_indexes:
            continue
        shared_target_indexes = target_indexes & student_indexes
        if not shared_target_indexes:
            continue
        for target_index in shared_target_indexes:
            _accumulate_pair_wls_stats(
                stats[:, source_index, target_index],
                source_model,
                student_indexes_by_node[target_index],
                max_wls_weight,
            )


def _accumulate_pair_wls_stats(
    stats: NDArray[np.float64],
    source_model: dict[str, float],
    target_model: dict[str, float],
    max_wls_weight: float,
):
    source_mean = source_model["mean"]
    target_mean = target_model["mean"]
    source_std = source_model["std"]
    target_std = target_model["std"]
    stats[_STATS_COUNT] += 1.0
    if not (
        math.isfinite(source_mean)
        and math.isfinite(target_mean)
        and math.isfinite(source_std)
        and math.isfinite(target_std)
    ):
        stats[_STATS_INVALID_INPUT_COUNT] += 1.0
        return

    uncertainty = source_std * source_std + target_std * target_std
    if not math.isfinite(uncertainty) or uncertainty <= 0.0:
        stats[_STATS_INVALID_UNCERTAINTY_COUNT] += 1.0
        return

    weight = min(1.0 / uncertainty, max_wls_weight)
    stats[_STATS_SOURCE_SUM] += source_mean
    stats[_STATS_SOURCE_SQUARE_SUM] += source_mean * source_mean
    stats[_STATS_WEIGHT_SUM] += weight
    stats[_STATS_WEIGHTED_SOURCE_SUM] += weight * source_mean
    stats[_STATS_WEIGHTED_TARGET_SUM] += weight * target_mean
    stats[_STATS_WEIGHTED_SOURCE_SQUARE_SUM] += weight * source_mean * source_mean
    stats[_STATS_WEIGHTED_SOURCE_TARGET_SUM] += weight * source_mean * target_mean


def _get_allowed_pair_indexes(
    node_to_index: dict[str, int],
    allowed_node_pairs: set[tuple[str, str]],
) -> list[tuple[int, int]]:
    pair_indexes = set()
    for source_node, target_node in allowed_node_pairs:
        if source_node not in node_to_index or target_node not in node_to_index:
            continue
        source_index = node_to_index[source_node]
        target_index = node_to_index[target_node]
        pair_indexes.add((source_index, target_index))
        pair_indexes.add((target_index, source_index))
    return sorted(pair_indexes)


def _get_allowed_target_indexes_by_source(
    allowed_pair_indexes: list[tuple[int, int]],
) -> dict[int, frozenset[int]]:
    target_indexes_by_source: dict[int, set[int]] = {}
    for source_index, target_index in allowed_pair_indexes:
        target_indexes_by_source.setdefault(source_index, set()).add(target_index)
    return {
        source_index: frozenset(target_indexes)
        for source_index, target_indexes in target_indexes_by_source.items()
    }
