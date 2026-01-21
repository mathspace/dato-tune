import logging
import os
import pickle
from pathlib import Path

import pandas as pd

from utils import ColumnMapping


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.data_folder = Path(config.get("common", "data_folder"))
        self.result_folder = Path(config.get("common", "result_folder"))

    def load_latern_responses(self, add_default_values=False, **kwargs):
        prefix = (
            f"{self.config.get('common', 'lantern_responses_csv_files_prefix')}*.csv"
        )
        curriculum_id = self.config.getint("common", "curriculum_id")
        csv_files = sorted(self.data_folder.glob(prefix))
        logging.info(
            f"Loading {len(csv_files)} CSV files from {self.data_folder} with prefix {prefix}"
        )
        df = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files])
        logging.info(f"All {len(csv_files)} CSV files successfully loaded.")
        logging.info(f"Preprocessing resulted dataframe with {len(df)} rows.")
        df = preprocess_qa_df(
            df, curriculum_id, add_default_values=add_default_values, **kwargs
        )
        logging.info(f"Final dataframe has {len(df)} rows.")
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

        # todo: use dataloader to minimize string file names


def preprocess_qa_df(df, curriculum_id, add_default_values=True, **kwargs):
    df[ColumnMapping.score] = (df[ColumnMapping.result] == "CORRECT").astype(float)
    df[ColumnMapping.dummy] = 1
    df[ColumnMapping.completed_at] = pd.to_datetime(
        df[ColumnMapping.completed_at]
    ).dt.tz_localize(None)

    # use only the data from the first catalog
    df = df.loc[df[ColumnMapping.curriculum_id] == curriculum_id].copy()

    if add_default_values:
        df[ColumnMapping.difficulty] = kwargs.get("default_difficulty", 0.0)
        df[ColumnMapping.discrimination] = kwargs.get("default_discrimination", 1.0)
        df[ColumnMapping.mastery] = kwargs.get("default_mastery", 0.0)

    return df


def load_catalog_hierarchy(folder_name, level="question_id"):
    drop_cols = ["description", "code", "title", "text", "option"]
    grade = (
        pd.read_csv(os.path.join(folder_name, "grade.csv"))
        .rename(columns={"id": "grade_id", "sana_topic_id": "grade"})
        .drop(drop_cols, axis=1, errors="ignore")
    )
    grade_strand = (
        pd.read_csv(os.path.join(folder_name, "gradestrand.csv"))
        .rename(columns={"id": "grade_strand_id", "sana_topic_id": "grade_strand"})
        .drop(drop_cols, axis=1, errors="ignore")
    )
    outcome = (
        pd.read_csv(os.path.join(folder_name, "outcome.csv"))
        .rename(columns={"id": "outcome_id", "sana_topic_id": "outcome"})
        .drop(drop_cols, axis=1, errors="ignore")
    )

    skill = (
        pd.read_csv(os.path.join(folder_name, "skill.csv"))
        .rename(columns={"id": "skill_id", "sana_topic_id": "skill"})
        .drop(drop_cols, axis=1, errors="ignore")
    )

    question = pd.read_csv(os.path.join(folder_name, "question.csv")).rename(
        columns={"id": "question_id"}
    )[["question_id", "skill_id"]]

    hierarchy = (
        question.merge(
            skill, on="skill_id", how="left", validate="m:1", suffixes=("", "_")
        )
        .merge(outcome, on="outcome_id", how="left", validate="m:1", suffixes=("", "_"))
        .merge(
            grade_strand,
            on="grade_strand_id",
            how="left",
            validate="m:1",
            suffixes=("", "_"),
        )
        .merge(grade, on="grade_id", how="left", validate="m:1", suffixes=("", "_"))
    )
    hierarchy.drop(
        [col for col in hierarchy.columns if col.endswith("_")], axis=1, inplace=True
    )
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
    hierarchy = hierarchy[sorted_cols[idx:]].drop_duplicates()
    logging.info(f"load_hierarchy: loading curriculum hierarchy from {folder_name}")

    return hierarchy


def load_snapshot_hierarchy(folder_name):
    knowledgegraph_snapshot = pd.read_csv(
        os.path.join(folder_name, "knowledgegraph_snapshot.csv")
    )
    checkin = pd.read_csv(os.path.join(folder_name, "checkin.csv"))

    snapshot_hierarchy = knowledgegraph_snapshot.rename(
        columns={"id": "knowledge_graph_snapshot_id"}
    ).merge(
        checkin.rename(columns={"id": "check_in_id", "user_id": "student_id"}),
        on="check_in_id",
        how="left",
        validate="1:1",
    )
    snapshot_hierarchy["started_at"] = pd.to_datetime(
        snapshot_hierarchy["started_at"]
    ).astype("<M8[ns]")
    snapshot_hierarchy["ended_at"] = pd.to_datetime(
        snapshot_hierarchy["ended_at"]
    ).astype("<M8[ns]")
    snapshot_hierarchy = snapshot_hierarchy.dropna()
    for col in snapshot_hierarchy.columns:
        if col.endswith("id"):
            snapshot_hierarchy[col] = snapshot_hierarchy[col].astype(int)
    return snapshot_hierarchy


def load_skill_snapshot(file_name):
    df = pd.read_csv(file_name).rename(columns={"id": "skill_snapshot_id"})
    for col in [
        "skill_snapshot_id",
        "knowledge_graph_snapshot_id",
        "skill_id",
        "true_proficiency",
        "true_proficiency_std",
    ]:
        assert col in df.columns, f"load_skill_snapshot: {col} not found in {file_name}"
    logging.info(f"load_skill_snapshot: loading skill snapshot  from {file_name}")
    return df


def load_teacher_difficulty(file_name):
    df = pd.read_csv(file_name)
    df.rename(
        columns={"Question": "question_id", "Question Difficulty": "difficulty"},
        inplace=True,
    )
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    df["skill_id"] = df["skill"].str.split("-", expand=True).iloc[:, 1]
    df["skill_id"] = df["skill_id"].astype(int)
    df["discrimination"] = 1.0
    logging.info(
        f"load_teacher_difficulty: loading teacher defined difficulty (cold start difficulty) from {file_name}"
    )
    return df


def load_estimated_difficulty(file_name):
    df = pd.read_csv(file_name, index_col=0)
    for col in ["difficulty", "discrimination", "question_id"]:
        assert col in df.columns, (
            f"load_estimated_difficulty: {col} not found in {file_name}!"
        )
    logging.info(
        f"load_estimated_difficulty: loading estimated difficulty from {file_name}"
    )
    return df


def load_estimated_mastery(file_name):
    df = pd.read_csv(file_name, index_col=0)
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
    # todo: this only loads a pickled object saved previously, should be done more properly
    skill_topics = pickle.load(open(file_name, "rb"))
    logging.info(f"load_knowledge_graph: loading knowledge graph from {file_name}")
    return skill_topics


def load_enriched_difficulty(file_name):
    df = pd.read_csv(file_name, index_col=0)
    for col in ["difficulty", "discrimination", "question_id"]:
        assert col in df.columns, (
            f"load_enriched_difficulty: {col} not found in {file_name}!"
        )
    logging.info(
        f"load_enriched_difficulty: loading enriched difficulty from {file_name}"
    )
    return df
