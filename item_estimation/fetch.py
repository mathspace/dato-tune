from __future__ import annotations

import logging
import os
from configparser import ConfigParser
from datetime import date
from textwrap import dedent
from typing import Literal, assert_type

import pandas as pd
import snowflake.connector


logger = logging.getLogger(__name__)


def _load_repo_config() -> ConfigParser:
    config = ConfigParser()
    config.read("config.ini")
    return config


def _get_snowflake_user() -> str:
    repo_config = _load_repo_config()
    username = repo_config.get("snowflake", "username", fallback="").strip()
    if username:
        return username

    raise RuntimeError(
        "Could not find a Snowflake username. Set [snowflake] username in config.ini."
    )


def _get_snowflake_password() -> str | None:
    return os.getenv("SNOWFLAKE_PASSWORD")


def _build_snowflake_connect_kwargs(region: Literal["au", "us"]) -> dict[str, str | bool]:
    account_name = "oua13326" if region == "us" else "pn30490.ap-southeast-2"
    return {
        "account": account_name,
        "user": _get_snowflake_user(),
        "role": "reporter",
        "warehouse": "reporting",
        "database": "data_science",
        "schema": "public",
    }


def _get_snowflake_connection(region: Literal["au", "us"]) -> snowflake.connector.SnowflakeConnection:
    connect_kwargs = _build_snowflake_connect_kwargs(region)
    try:
        return snowflake.connector.connect(
            **connect_kwargs,
            authenticator="externalbrowser",
            client_store_temporary_credential=True,
        )
    except Exception as browser_exc:
        password = _get_snowflake_password()
        if not password:
            raise RuntimeError(
                "Snowflake externalbrowser authentication failed and no fallback password "
                "was found. Export SNOWFLAKE_PASSWORD for username_password_mfa."
            ) from browser_exc

        logger.warning(
            "Snowflake externalbrowser authentication failed; falling back to username_password_mfa"
        )
        return snowflake.connector.connect(
            **connect_kwargs,
            authenticator="username_password_mfa",
            password=password,
            client_store_temporary_credential=True,
        )


def fetch_lantern_repsonses_range(
    curriculum_id: int,
    region: Literal["au", "us"],
    begin_date: date,
    end_date: date,
) -> pd.DataFrame:
    assert_type(begin_date, date)
    assert_type(end_date, date)
    query = dedent(f"""
        SELECT
            student_id,
            question_public_id,
            latest_question_version_id,
            question_version_id,
            grade_strand_id,
            grade_substrand_id,
            skill_id,
            cold_start_difficulty,
            result,
            created_at,
            curriculum_id,
        FROM DATA_SCIENCE.PREPROCESSING.LANTERN_RESPONSES
        WHERE created_at >= '{begin_date.isoformat()}'
            AND created_at <= '{end_date.isoformat()}'
            AND curriculum_id = '{curriculum_id}'
    """)

    return fetch_responses_from_snowflake(curriculum_id, region, query)

def _build_window_filter(max_windows: int | None, window_index: int | None) -> str:
    if window_index is not None:
        return f"\n            AND window_index = {window_index}"
    if max_windows is not None:
        return f"\n            AND window_index < {max_windows}"
    return ""


# We want to maintain data localised in time. Student ability is expected to change over
# time, so we can't expect a single student's data ranging over long period of time
# to reliably estimate their abilty. However, we don't want to ignore a
# significant amount of usable data by only considering a small period per student.

# We use a 'windowed' approach where each students activity is chunked into periods,
# and for the purposes of estimation each student-window is a unique agent
# with distinct topic-abilities.
def _build_lantern_windowed_query(
    curriculum_id: int,
    window_size_months: int,
    max_windows: int | None = None,
    window_index: int | None = None,
) -> str:
    window_filter = _build_window_filter(max_windows, window_index)
    return dedent(f"""
        WITH earliest_date AS (
            SELECT MIN(created_at) as min_date
            FROM DATA_SCIENCE.PREPROCESSING.LANTERN_RESPONSES
            WHERE curriculum_id = '{curriculum_id}'
        ),
        date_sequence AS (
            SELECT
                ROW_NUMBER() OVER (ORDER BY SEQ4()) - 1 as window_index
            FROM TABLE(GENERATOR(ROWCOUNT => 500))
        ),
        windows AS (
            SELECT
                window_index,
                DATEADD(month, -{window_size_months} * window_index, CURRENT_DATE()) as window_end,
                DATEADD(month, -{window_size_months}, DATEADD(month, -{window_size_months} * window_index, CURRENT_DATE())) as window_start
            FROM date_sequence, earliest_date
            WHERE window_start >= min_date{window_filter}
        )
        SELECT
            lr.student_id,
            lr.question_public_id,
            lr.latest_question_version_id,
            lr.question_version_id,
            lr.grade_strand_id,
            lr.grade_substrand_id,
            lr.skill_id,
            lr.cold_start_difficulty,
            lr.result,
            lr.created_at,
            lr.curriculum_id,
            w.window_index
        FROM DATA_SCIENCE.PREPROCESSING.LANTERN_RESPONSES lr
        INNER JOIN windows w
            ON lr.created_at >= w.window_start
            AND lr.created_at < w.window_end
        WHERE lr.curriculum_id = '{curriculum_id}'
    """)


def fetch_lantern_responses_windowed(
    curriculum_id: int,
    region: Literal["au", "us"],
    window_size_months: int,
    max_windows: int | None = None,
    window_index: int | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> pd.DataFrame:
    """
    Fetch lantern responses with sliding window indices.

    Uses configurable window size with same stride, going back from today.

    Args:
        curriculum_id: The curriculum ID to filter by
        region: Region for Snowflake account ('us' or 'au')
        window_size_months: Size of each window in months
        max_windows: If set, only fetch this many windows (most recent first)
        window_index: If set, only fetch this specific window index

    Returns:
        DataFrame with responses, where each response may appear in multiple windows.
        Includes window_index column (0 = most recent window, 1 = next window back, etc.)
    """
    query = _build_lantern_windowed_query(curriculum_id, window_size_months, max_windows, window_index)
    return fetch_responses_from_snowflake(curriculum_id, region, query, limit, offset)


def _build_mathspace_windowed_query(
    curriculum_id: int,
    window_size_months: int,
    max_windows: int | None = None,
    window_index: int | None = None,
) -> str:
    window_filter = _build_window_filter(max_windows, window_index)
    return dedent(f"""
        WITH earliest_date AS (
            SELECT MIN(completed_at) as min_date
            FROM DATA_SCIENCE.PREPROCESSING.MATHSPACE_RESPONSES
            WHERE curriculum_id = '{curriculum_id}'
        ),
        date_sequence AS (
            SELECT
                ROW_NUMBER() OVER (ORDER BY SEQ4()) - 1 as window_index
            FROM TABLE(GENERATOR(ROWCOUNT => 500))
        ),
        windows AS (
            SELECT
                window_index,
                DATEADD(month, -{window_size_months} * window_index, CURRENT_DATE()) as window_end,
                DATEADD(month, -{window_size_months}, DATEADD(month, -{window_size_months} * window_index, CURRENT_DATE())) as window_start
            FROM date_sequence, earliest_date
            WHERE window_start >= min_date{window_filter}
        )
        SELECT
            mr.user_id AS student_id,
            mr.problem_item_id AS question_version_id,
            mr.problem_item_id AS latest_question_version_id,
            mr.problem_item_id AS question_public_id,
            mr.gradesubstrand_id AS grade_substrand_id,
            mr.gradestrand_id AS grade_strand_id,
            mr.skill_id,
            CASE WHEN mr.score = 1 THEN 'CORRECT' ELSE 'INCORRECT' END AS result,
            mr.completed_at AS created_at,
            mr.difficulty AS cold_start_difficulty,
            mr.discrimination,
            mr.curriculum_id,
            w.window_index
        FROM DATA_SCIENCE.PREPROCESSING.MATHSPACE_RESPONSES mr
        INNER JOIN windows w
            ON mr.completed_at >= w.window_start
            AND mr.completed_at < w.window_end
        WHERE mr.curriculum_id = '{curriculum_id}'
            AND mr.completed_at > mr.problem_template_updated_at
    """)


def fetch_mathspace_responses_windowed(
    curriculum_id: int,
    region: Literal["au", "us"],
    window_size_months: int,
    max_windows: int | None = None,
    window_index: int | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> pd.DataFrame:
    query = _build_mathspace_windowed_query(curriculum_id, window_size_months, max_windows, window_index)
    return fetch_responses_from_snowflake(curriculum_id, region, query, limit, offset)


def fetch_responses_from_snowflake(
    curriculum_id: int,
    region: Literal["au", "us"],
    query: str,
    limit: int | None = None,
    offset: int = 0,
) -> pd.DataFrame:
    assert_type(curriculum_id, int)
    if limit is not None:
        query = f"SELECT * FROM ({query.strip()}) LIMIT {limit} OFFSET {offset}"
    conn = _get_snowflake_connection(region)
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        return cursor.fetch_pandas_all()
    finally:
        conn.close()
