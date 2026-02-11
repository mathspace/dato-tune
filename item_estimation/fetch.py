from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from datetime import date
from textwrap import dedent
from typing import Literal, assert_type

import pandas as pd


logger = logging.getLogger(__name__)

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
            question_version_id,
            grade_strand_id,
            grade_substrand_id,
            skill_id,
            cold_start_difficulty,
            result,
            created_at,
            curriculum_id
        FROM DATA_SCIENCE.PREPROCESSING.LANTERN_RESPONSES
        WHERE created_at >= '{begin_date.isoformat()}'
            AND created_at <= '{end_date.isoformat()}'
            AND curriculum_id = '{curriculum_id}'
    """)

    return fetch_lantern_responses_from_snowflake(curriculum_id, region, query)

def fetch_lantern_responses_windowed(
    curriculum_id: int,
    region: Literal["au", "us"],
) -> pd.DataFrame:
    """
    Fetch lantern responses with sliding window indices.

    Uses fixed 12-month windows with 6-month stride, going back from today.
    Each response may appear in multiple windows (up to 2 consecutive windows due to overlap).

    Args:
        curriculum_id: The curriculum ID to filter by
        region: Region for Snowflake account ('us' or 'au')

    Returns:
        DataFrame with responses, where each response may appear in multiple windows.
        Includes window_index column (0 = most recent 12 months, 1 = next 12 months back, etc.)
    """

    query = dedent(f"""
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
                DATEADD(month, -6 * window_index, CURRENT_DATE()) as window_end,
                DATEADD(month, -12, DATEADD(month, -6 * window_index, CURRENT_DATE())) as window_start
            FROM date_sequence, earliest_date
            WHERE window_start >= min_date
        )
        SELECT
            lr.student_id,
            lr.question_public_id,
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

    return fetch_lantern_responses_from_snowflake(curriculum_id, region, query)

def fetch_lantern_responses_from_snowflake(
    curriculum_id: int,
    region: Literal["au", "us"],
    query: str,
) -> pd.DataFrame:
    
    assert_type(curriculum_id, int)

    account_name = "oua13326" if region == "us" else "pn30490.ap-southeast-2"


    if shutil.which("snowsql") is None:
        raise RuntimeError(
            "snowsql not found - please install SnowSQL from https://docs.snowflake.com/en/user-guide/snowsql-install-config.html"
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as temp_outfile:
        try:
            cmd = [
                "snowsql",
                "--accountname",
                account_name,
                "--authenticator",
                "externalbrowser",
                "--warehouse",
                "reporting",
                "--dbname",
                "data_science",
                "--schemaname",
                "public",
                "--option",
                "output_format=csv",
                "--option",
                "header=true",
                "--option",
                "timing=false",
                "--option",
                "friendly=false",
                "--option",
                f"output_file={temp_outfile.name}",
                "--query",
                query,
            ]

            logging.info(
                "Fetching data from Snowflake... (web browser will open for authentication)"
            )
            _ = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to fetch data from Snowflake.\n\nSTDERR: {e.stderr}\n\nSTDOUT: {e.output}"
            )

        try:
            df = pd.read_csv(temp_outfile.name)
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to process Snowflake data.\nError: {e}")
