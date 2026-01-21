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


def fetch_lantern_responses_from_snowflake(
    curriculum_id: int, region: Literal["au", "us"], begin_date: date, end_date: date
) -> pd.DataFrame:
    assert_type(curriculum_id, int)
    assert_type(begin_date, date)
    assert_type(end_date, date)

    account_name = "oua13326" if region == "us" else "pn30490.ap-southeast-2"

    query = dedent(f"""
        SELECT
            student_id,
            question_public_id,
            grade_strand_id,
            cold_start_difficulty,
            result,
            created_at,
            curriculum_id
        FROM DATA_SCIENCE.PREPROCESSING.LANTERN_RESPONSES
        WHERE created_at >= '{begin_date.isoformat()}'
            AND created_at <= '{end_date.isoformat()}'
            AND curriculum_id = '{curriculum_id}'
    """)

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
