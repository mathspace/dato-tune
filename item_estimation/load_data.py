import logging
import os
import pickle
import hashlib
from pathlib import Path

import polars as pl

from utils import ColumnMapping


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.data_folder = Path(config.get("common", "data_folder"))
        self.result_folder = Path(config.get("common", "result_folder"))

    def load_latern_responses(
        self, df: pl.DataFrame, curriculum_id: int, add_default_values=False, **kwargs
    ):
        df = preprocess_qa_df(
            df, curriculum_id, add_default_values=add_default_values, **kwargs
        )
        return df

    def catalog_hierarchy(self, level="question_id"):
        return load_catalog_hierarchy(self.data_folder, level=level)

    def snapshot_hierarchy(self):
        return load_snapshot_hierarchy(self.data_folder)

    def skill_snapshot(self):
        return load_skill_snapshot(os.path.join(self.data_folder, "skill_snapshot.csv"))

    def teacher_difficulty(self):
        return load_teacher_difficulty(
            os.path.join(self.data_folder, "ac-questions.csv")
        )

    def estimated_difficulty(self):
        return load_estimated_difficulty(
            os.path.join(self.result_folder, "estimated_item.csv")
        )

    def estimated_mastery(self):
        return load_estimated_mastery(
            os.path.join(self.result_folder, "estimated_mastery.csv")
        )

    def knowledge_graph(self):
        return load_knowledge_graph(os.path.join(self.data_folder, "skill_topics.p"))

    def enriched_difficulty(self):
        return load_enriched_difficulty(
            os.path.join(self.result_folder, "enriched_difficulty.csv")
        )


def preprocess_qa_df(
    df: pl.DataFrame,
    curriculum_id: int,
    add_default_values: bool = True,
    student_sample_rate: float | None = None,
    **kwargs,
) -> pl.DataFrame:
    df = df.with_columns([
        (pl.col(ColumnMapping.result) == "CORRECT").cast(pl.Float64).alias(ColumnMapping.score),
        pl.lit(1).alias(ColumnMapping.dummy),
        pl.col(ColumnMapping.completed_at)
            .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%:z", strict=False)
            .dt.replace_time_zone(None)
            .alias(ColumnMapping.completed_at),
    ])
    df = df.filter(pl.col(ColumnMapping.curriculum_id) == curriculum_id)
    df = maybe_sample_students(df, student_sample_rate)

    if add_default_values:
        df = df.with_columns([
            pl.lit(kwargs.get("default_difficulty", 0.0)).cast(pl.Float64).alias(ColumnMapping.difficulty),
            pl.lit(kwargs.get("default_discrimination", 1.0)).cast(pl.Float64).alias(ColumnMapping.discrimination),
            pl.lit(kwargs.get("default_mastery", 0.0)).cast(pl.Float64).alias(ColumnMapping.mastery),
        ])

    return df


def maybe_sample_students(
    df: pl.DataFrame, student_sample_rate: float | None = None
) -> pl.DataFrame:
    if student_sample_rate is None or student_sample_rate == 1.0:
        return df
    if not 0.0 <= student_sample_rate <= 1.0:
        raise ValueError(
            f"student_sample_rate must be between 0 and 1 inclusive, got {student_sample_rate}"
        )
    if df.is_empty():
        return df

    unique_students = df.select(ColumnMapping.student_id).unique()
    sampled_students = unique_students.with_columns(
        pl.col(ColumnMapping.student_id)
        .cast(pl.Utf8)
        .map_elements(_student_sample_score, return_dtype=pl.Float64)
        .alias("_student_sample_score")
    ).filter(pl.col("_student_sample_score") < student_sample_rate)

    logging.info(
        "student sampling enabled at %.2f: kept %s/%s students",
        student_sample_rate,
        f"{sampled_students.height:,}",
        f"{unique_students.height:,}",
    )
    return df.join(
        sampled_students.select(ColumnMapping.student_id),
        on=ColumnMapping.student_id,
        how="inner",
    )


def _student_sample_score(student_id: str) -> float:
    digest = hashlib.sha256(student_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def load_catalog_hierarchy(folder_name, level="question_id"):
    drop_cols = ["description", "code", "title", "text", "option"]

    def _read_and_rename(fname, id_col, sana_col):
        df = pl.read_csv(os.path.join(folder_name, fname)).rename({
            "id": id_col, "sana_topic_id": sana_col
        })
        return df.drop([c for c in drop_cols if c in df.columns])

    grade = _read_and_rename("grade.csv", "grade_id", "grade")
    grade_strand = _read_and_rename("gradestrand.csv", "grade_strand_id", "grade_strand")
    outcome = _read_and_rename("outcome.csv", "outcome_id", "outcome")
    skill = _read_and_rename("skill.csv", "skill_id", "skill")

    question = (
        pl.read_csv(os.path.join(folder_name, "question.csv"))
        .rename({"id": "question_id"})
        .select(["question_id", "skill_id"])
    )

    hierarchy = (
        question
        .join(skill, on="skill_id", how="left", suffix="_r")
        .join(outcome, on="outcome_id", how="left", suffix="_r")
        .join(grade_strand, on="grade_strand_id", how="left", suffix="_r")
        .join(grade, on="grade_id", how="left", suffix="_r")
    )
    # Drop any duplicate-suffix columns produced by joins
    hierarchy = hierarchy.drop([c for c in hierarchy.columns if c.endswith("_r")])

    sorted_cols = [
        "question_id",
        "skill_id",
        "outcome_id",
        "grade_strand_id",
        "grade_id",
        "strand_id",
        "curriculum_id",
    ]
    if level not in sorted_cols:
        idx = 0
    else:
        idx = sorted_cols.index(level)
    cols = [c for c in sorted_cols[idx:] if c in hierarchy.columns]
    hierarchy = hierarchy.select(cols).unique()
    logging.info(f"load_hierarchy: loading curriculum hierarchy from {folder_name}")
    return hierarchy


def load_snapshot_hierarchy(folder_name):
    knowledgegraph_snapshot = pl.read_csv(
        os.path.join(folder_name, "knowledgegraph_snapshot.csv")
    ).rename({"id": "knowledge_graph_snapshot_id"})

    checkin = pl.read_csv(
        os.path.join(folder_name, "checkin.csv")
    ).rename({"id": "check_in_id", "user_id": "student_id"})

    snapshot_hierarchy = (
        knowledgegraph_snapshot
        .join(checkin, on="check_in_id", how="left")
        .drop_nulls()
    )

    snapshot_hierarchy = snapshot_hierarchy.with_columns([
        pl.col("started_at").str.to_datetime(format=None, strict=False).cast(pl.Datetime("ns")).dt.replace_time_zone(None),
        pl.col("ended_at").str.to_datetime(format=None, strict=False).cast(pl.Datetime("ns")).dt.replace_time_zone(None),
    ])

    id_cols = [c for c in snapshot_hierarchy.columns if c.endswith("id")]
    snapshot_hierarchy = snapshot_hierarchy.with_columns([
        pl.col(c).cast(pl.Int64) for c in id_cols
    ])
    return snapshot_hierarchy


def load_skill_snapshot(file_name):
    df = pl.read_csv(file_name).rename({"id": "skill_snapshot_id"})
    for col in [
        "skill_snapshot_id",
        "knowledge_graph_snapshot_id",
        "skill_id",
        "true_proficiency",
        "true_proficiency_std",
    ]:
        assert col in df.columns, f"load_skill_snapshot: {col} not found in {file_name}"
    logging.info(f"load_skill_snapshot: loading skill snapshot from {file_name}")
    return df


def load_teacher_difficulty(file_name):
    df = (
        pl.read_csv(file_name)
        .rename({"Question": "question_id", "Question Difficulty": "difficulty"})
    )
    df = df.rename({c: c.lower().replace(" ", "_") for c in df.columns})
    df = df.with_columns([
        pl.col("skill").str.split("-").list.get(1).cast(pl.Int64).alias("skill_id"),
        pl.lit(1.0).alias("discrimination"),
    ])
    logging.info(
        f"load_teacher_difficulty: loading teacher defined difficulty (cold start difficulty) from {file_name}"
    )
    return df


def load_estimated_difficulty(file_name):
    df = pl.read_csv(file_name)
    for col in ["difficulty", "discrimination", "question_id"]:
        assert col in df.columns, (
            f"load_estimated_difficulty: {col} not found in {file_name}!"
        )
    logging.info(
        f"load_estimated_difficulty: loading estimated difficulty from {file_name}"
    )
    return df


def load_estimated_mastery(file_name):
    df = pl.read_csv(file_name)
    cols = list(df.columns)
    for col in ["mastery", "student_id"]:
        assert col in cols, f"load_estimated_mastery: {col} not found in {file_name}!"
        cols.remove(col)
    if len(cols) > 0:
        logging.info(
            f"{cols} should be the granularity of mastery estimation, please verify."
        )
    logging.info(f"load_estimated_mastery: loading estimated mastery from {file_name}")
    return df


def load_knowledge_graph(file_name):
    skill_topics = pickle.load(open(file_name, "rb"))
    logging.info(f"load_knowledge_graph: loading knowledge graph from {file_name}")
    return skill_topics


def load_enriched_difficulty(file_name):
    df = pl.read_csv(file_name)
    for col in ["difficulty", "discrimination", "question_id"]:
        assert col in df.columns, (
            f"load_enriched_difficulty: {col} not found in {file_name}!"
        )
    logging.info(
        f"load_enriched_difficulty: loading enriched difficulty from {file_name}"
    )
    return df
