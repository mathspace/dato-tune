from collections.abc import Callable, Iterable

from propagation_parameters_estimation.bayes_irt import numba_event_update, sigmoid

UNIFORM_PRIOR = {
    "mean": 0.0,
    "std": 100.0,
}

ResponseRow = tuple[str, str, float, float, int]


def update_student_model(
    student_node_models: dict[str, dict[str, float]],
    node: str,
    difficulty: float,
    response: int,
    discrimination: float,
):
    if node not in student_node_models:
        student_node_models[node] = UNIFORM_PRIOR.copy()
    assert discrimination > 0.0
    update = numba_event_update(
        ability=sigmoid(student_node_models[node]["mean"]),
        ability_sigma=student_node_models[node]["std"],
        difficulty=sigmoid(difficulty),
        discriminative_index=discrimination,
        response=bool(response),
    )
    student_node_models[node]["mean"] = float(update[0])
    student_node_models[node]["std"] = float(update[1])


def get_student_estimation_models(
    response_rows: Iterable[ResponseRow],
    progress_callback: Callable[[int, int], None] | None = None,
    progress_interval: int | None = None,
    total_responses: int | None = None,
):
    student_estimation_models: dict[str, dict[str, dict[str, float]]] = {}
    responses_processed = 0
    last_progress_report = 0
    next_progress_report = progress_interval if progress_interval else None
    for responses_processed, (
        student_id,
        node,
        difficulty,
        discrimination,
        response,
    ) in enumerate(response_rows, start=1):
        student_node_models = student_estimation_models.get(student_id)
        if student_node_models is None:
            student_node_models = {}
            student_estimation_models[student_id] = student_node_models
        update_student_model(
            student_node_models,
            node,
            float(difficulty),
            int(response),
            float(discrimination),
        )
        if (
            progress_callback is not None
            and next_progress_report is not None
            and responses_processed >= next_progress_report
        ):
            progress_callback(responses_processed, len(student_estimation_models))
            last_progress_report = responses_processed
            next_progress_report += progress_interval

    if (
        progress_callback is not None
        and total_responses is not None
        and responses_processed == total_responses
        and last_progress_report != responses_processed
    ):
        progress_callback(responses_processed, len(student_estimation_models))

    return student_estimation_models
