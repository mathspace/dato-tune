import logging
import os
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path
from typing import Dict, TextIO

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import auc, roc_curve

import model_inference as mi
from load_data import DataLoader
from model_inference import logistic_cdf
from utils import ColumnMapping


def load_inference_data(data_loader: DataLoader):
    df = data_loader.load_latern_responses()
    return df


def run_mle(
    train_data: pl.DataFrame,
    granularity,
    infer_mastery=True,
    infer_item=True,
    tune_discrimination=False,
    n_iter=30,
    tol=0.01,
):
    df = train_data
    # add default value as initial value of optimisation
    for col, default_value in zip(
        [ColumnMapping.mastery, ColumnMapping.difficulty, ColumnMapping.discrimination],
        [0.0, 0.0, 1.0],
    ):
        if col not in df.columns:
            df = df.with_columns(pl.lit(default_value).cast(pl.Float64).alias(col))

    # Select columns including window_col if provided
    cols_to_keep = [
        ColumnMapping.student_id,
        granularity,
        ColumnMapping.estimate_question_id,
        ColumnMapping.score,
        ColumnMapping.difficulty,
        ColumnMapping.discrimination,
        ColumnMapping.mastery,
    ]
    using_window_col = ColumnMapping.window_index in df.columns
    if using_window_col:
        cols_to_keep.append(ColumnMapping.window_index)

    df = df.select(cols_to_keep)

    # Track initial item count for reporting
    total_items = df[ColumnMapping.estimate_question_id].n_unique()
    logging.info(f"Starting optimization with {total_items} unique items")

    n_obs = len(df)
    likelihood = mi.total_likelihood(df)
    avg_likelihood = likelihood / n_obs
    estimation_tracking = [(0, likelihood, n_obs, avg_likelihood)]
    for it in range(n_iter):
        if infer_item:
            df = mi.batch_item_estimation(df, tune_discrimination=tune_discrimination)
            n_items = df[ColumnMapping.estimate_question_id].n_unique()
            item_pct = 100 * n_items / total_items
            logging.info(
                f"iteration: {it}, step: item estimation, items: {n_items}/{total_items} ({item_pct:.1f}%)"
            )

        if infer_mastery:
            df = mi.batch_mastery_estimation(df, granularity_col=granularity, using_window_col=using_window_col)
            logging.info(f"iteration: {it}, step: mastery estimation, n_obs: {len(df):,}")

        n_obs = len(df)
        likelihood = mi.total_likelihood(df)
        avg_likelihood = likelihood / n_obs
        estimation_tracking.append((it + 1, likelihood, n_obs, avg_likelihood))
        logging.info(
            f"iteration: {it}, total likelihood: {likelihood}, n_obs: {n_obs}, avg: {avg_likelihood:.6f}"
        )

        if len(estimation_tracking) >= 3:
            prev_avg = estimation_tracking[-2][3]
            curr_avg = estimation_tracking[-1][3]
            relative_benefit = (curr_avg - prev_avg) / abs(prev_avg)
            if 0 < relative_benefit < tol:
                logging.info(
                    f"optimisation stopped at iteration {it}, improvement: {relative_benefit:.4%}, tolerance: {tol:.4%}"
                )
                break

    df = df.with_columns(
        pl.Series(
            name=ColumnMapping.p_correct,
            values=mi.p_correct(
                df[ColumnMapping.mastery].to_numpy().astype(np.float64),
                df[ColumnMapping.difficulty].to_numpy().astype(np.float64),
                df[ColumnMapping.discrimination].to_numpy().astype(np.float64),
            ),
        )
    )
    estimation_tracking = pl.DataFrame(
        data=estimation_tracking,
        schema={"iter": pl.Int64, "likelihood": pl.Float64, "n_obs": pl.Int64, "avg_likelihood": pl.Float64},
        orient="row",
    )
    return estimation_tracking, df


def mle_track_plot(
    tracking: pl.DataFrame, title="Likelihood per iteration", file_name=None, display=True
):
    p = sns.lineplot(x="iter", y="likelihood", data=tracking)
    p.set_title(title)
    if file_name:
        plt.savefig(file_name)
        logging.info(f"mle_track_plot: {title} saved as {file_name}")
    if display:
        plt.show()
    return


def roc_plot(df: pl.DataFrame, title="ROC curve", file_name=None, display=True):
    fpr, tpr, _ = roc_curve(
        df[ColumnMapping.score].to_numpy(),
        df[ColumnMapping.p_correct].to_numpy(),
    )
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(
        fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    if file_name:
        plt.savefig(file_name)
        logging.info(f"roc_plot: {title} saved as {file_name}")
    if display:
        plt.show()
    return roc_auc


def estimation_histogram(df: pl.DataFrame, title="Histogram", file_name=None, display=True):
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(18, 5))
    ax[0].hist(df[ColumnMapping.difficulty].to_numpy(), bins=30)
    ax[0].set_title("Estimated difficulty")

    ax[1].hist(df[ColumnMapping.discrimination].to_numpy(), bins=30)
    ax[1].set_title("Estimated discrimination index")

    ax[2].hist(df[ColumnMapping.mastery].to_numpy(), bins=30)
    ax[2].set_title("Estimated mastery")

    plt.suptitle(title, fontsize=14)
    if file_name:
        plt.savefig(file_name)
        logging.info(f"estimation_histogram: {title} saved as {file_name}")
    if display:
        plt.show()
    return


def get_result(
    train_result: pl.DataFrame,
    granularity,
    original_questions_dificulties: Dict[str, dict],
    file_path=None,
    outfile_suffix: str | None = None,
    using_window_col=False,
):
    group_cols = [ColumnMapping.student_id, granularity]
    if using_window_col:
        group_cols.append(ColumnMapping.window_index)

    estimated_mastery = train_result.group_by(group_cols).agg([
        pl.col(ColumnMapping.mastery).mean(),
        pl.col(ColumnMapping.estimate_question_id).n_unique().alias("n_question"),
    ])

    estimated_difficulty = train_result.group_by(ColumnMapping.estimate_question_id).agg([
        pl.col(ColumnMapping.difficulty).mean(),
        pl.col(ColumnMapping.discrimination).mean(),
        pl.col(ColumnMapping.student_id).n_unique().alias("n_student"),
    ])

    # Look up original difficulty by version_id (only exists for latest versions)
    estimated_difficulty = estimated_difficulty.with_columns(
        pl.col(ColumnMapping.estimate_question_id)
        .map_elements(
            lambda v_id: original_questions_dificulties.get(v_id, {}).get("difficulty"),
            return_dtype=pl.Float64,
        )
        .alias("OriginalDifficulty")
    )

    # Filter to only items with original difficulty (i.e., latest versions)
    estimated_difficulty_latest = estimated_difficulty.filter(
        pl.col("OriginalDifficulty").is_not_null()
    )
    estimated_difficulty_latest = estimated_difficulty_latest.with_columns(
        pl.Series(
            name="CalibratedDifficulty",
            values=logistic_cdf(
                estimated_difficulty_latest[ColumnMapping.difficulty].to_numpy().astype(np.float64)
            ),
        )
    )
    estimated_difficulty_latest = estimated_difficulty_latest.with_columns(
        (pl.col("CalibratedDifficulty") - pl.col("OriginalDifficulty"))
        .abs()
        .alias("DifficultiesDifference abs(Original-Calibrated)")
    )

    if outfile_suffix:
        with open(f"difficulties_{outfile_suffix}.csv", "w") as f:
            estimated_difficulty_latest.write_csv(f)
            logging.info("estimated difficulty saved to outfile")

    if file_path:
        mastery_file = os.path.join(file_path, "estimated_mastery.csv")
        difficulty_file = os.path.join(file_path, "estimated_item.csv")
        estimated_mastery.write_csv(mastery_file)
        estimated_difficulty_latest.write_csv(difficulty_file)
        logging.info(
            f"estimated mastery and difficulty are saved as files {mastery_file} and {difficulty_file}"
        )

    return estimated_mastery, estimated_difficulty


def calc_test_result(
    estimated_mastery: pl.DataFrame,
    estimated_difficulty: pl.DataFrame,
    test_data: pl.DataFrame,
    granularity,
    using_window_col=False,
):
    merge_cols = [ColumnMapping.student_id, granularity]
    if using_window_col:
        merge_cols.append(ColumnMapping.window_index)

    cols_to_drop = [
        c for c in [ColumnMapping.difficulty, ColumnMapping.discrimination]
        if c in test_data.columns
    ]
    df = (
        test_data.drop(cols_to_drop)
        .join(estimated_mastery, on=merge_cols, how="inner")
        .join(estimated_difficulty, on=ColumnMapping.estimate_question_id, how="inner")
    )
    df = df.with_columns(
        pl.Series(
            name=ColumnMapping.p_correct,
            values=mi.p_correct(
                df[ColumnMapping.mastery].to_numpy().astype(np.float64),
                df[ColumnMapping.difficulty].to_numpy().astype(np.float64),
                df[ColumnMapping.discrimination].to_numpy().astype(np.float64),
            ),
        )
    )
    logging.info(
        f"{np.round(df.shape[0] / test_data.shape[0], 2)} of test data has estimated mastery and difficulty"
    )
    return df


def benchmark_gbm(train_data: pl.DataFrame, test_data: pl.DataFrame, **kwargs):
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import GridSearchCV

    show_graph = kwargs.get("display", kwargs.get("show_graph", False))
    train_data = train_data.with_columns(
        (pl.col(ColumnMapping.estimate_question_id).rank(method="dense").cast(pl.Int32) - 1)
        .alias("question_num_id")
    )
    test_df_sub = test_data.join(
        train_data.select(ColumnMapping.student_id).unique(),
        on=ColumnMapping.student_id,
        how="inner",
    ).join(
        train_data.select([ColumnMapping.estimate_question_id, "question_num_id"]).unique(),
        on=ColumnMapping.estimate_question_id,
        how="inner",
    )
    X_train = train_data.select([ColumnMapping.student_id, "question_num_id"]).to_numpy()
    y_train = train_data[ColumnMapping.score].to_numpy().astype(np.float64)

    seed = kwargs.get("random_seed", 124)
    np.random.seed(seed)
    parameters = kwargs.get(
        "parameters", {"max_depth": np.arange(1, 8, 2), "n_estimators": [50]}
    )
    logging.info(f"GBM benchmark: search on parameters {parameters}")
    gbm = LGBMClassifier()
    clf = GridSearchCV(gbm, parameters, scoring="roc_auc", cv=3, n_jobs=4)
    clf.fit(X_train, y_train)
    logging.info(f"GBM benchmark: best parameters {clf.best_estimator_.get_params()}")

    model = LGBMClassifier(**clf.best_estimator_.get_params())
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_train)[:, 1]
    auc_train = roc_plot(
        pl.DataFrame({ColumnMapping.p_correct: proba, ColumnMapping.score: y_train}),
        "GBM training ROC",
        file_name=kwargs.get("train_roc_file", None),
        display=show_graph,
    )
    logging.info(f"Benchmark: training ROC AUC score is {auc_train}")

    X_test = test_df_sub.select([ColumnMapping.student_id, "question_num_id"]).to_numpy()
    y_test = test_df_sub[ColumnMapping.score].to_numpy().astype(np.float64)
    proba = model.predict_proba(X_test)[:, 1]
    auc_test = roc_plot(
        pl.DataFrame({ColumnMapping.p_correct: proba, ColumnMapping.score: y_test}),
        "GBM testing ROC",
        file_name=kwargs.get("train_roc_file", None),
        display=show_graph,
    )
    logging.info(f"Benchmark: testing ROC AUC score is {auc_test}")

    return model


def get_questions_difficulties(df: pl.DataFrame) -> Dict[str, dict]:
    latest_versions = df.filter(
        pl.col(ColumnMapping.estimate_question_id) == pl.col(ColumnMapping.latest_question_version_id)
    )
    return {
        row[0]: {"public_id": row[1], "difficulty": row[2]}
        for row in latest_versions.select([
            ColumnMapping.estimate_question_id,
            ColumnMapping.question_public_id,
            ColumnMapping.difficulty,
        ]).unique().iter_rows()
    }


def run(config: ConfigParser, df: pl.DataFrame, outfile_suffix: str):
    inference_config = config["inference"]

    result_folder = Path(inference_config["result_folder"], outfile_suffix)
    result_folder.mkdir(exist_ok=True, parents=True)
    granularity = inference_config["granularity_col"]
    n_iter = inference_config.getint("n_iter", 15)
    tol = inference_config.getfloat("tol", 0.01)
    infer_mastery = inference_config.getboolean("infer_mastery", True)
    infer_item = inference_config.getboolean("infer_item", True)
    tune_discrimination = inference_config.getboolean("tune_discrimination", False)
    is_benchmark = inference_config.getboolean("is_benchmark", False)
    show_graph = inference_config.getboolean("show_graph", False)
    random_seed = inference_config.getint("random_seed", 123)
    min_obs = inference_config.getint("min_obs", 10)
    split_ratio = inference_config.getfloat("split_ratio", 0.2)

    np.random.seed(random_seed)

    granularity_col = getattr(ColumnMapping, granularity)
    if granularity_col is None:
        raise ValueError(f"granularity_col {granularity} is not a valid ColumnMapping attribute")

    qa_history = df

    group_cols = [ColumnMapping.student_id, granularity_col]
    using_window_col = ColumnMapping.window_index in qa_history.columns
    if using_window_col:
        group_cols.append(ColumnMapping.window_index)

    logging.info(f"filtering groups with fewer than {min_obs} observations...")
    qa_history = mi.remove_groups_with_insufficient_data(
        qa_history, group_cols, min_obs
    )
    logging.info(f"  {len(qa_history):,} rows remaining")
    logging.info("filtering groups with all incorrect responses...")
    qa_history = mi.remove_groups_with_all_incorrect(qa_history, group_cols)
    logging.info(f"  {len(qa_history):,} rows remaining")
    logging.info("filtering groups with all correct responses...")
    qa_history = mi.remove_groups_with_all_correct(qa_history, group_cols)
    logging.info(f"  {len(qa_history):,} rows remaining")

    logging.info("extracting question difficulties...")
    original_difficulties = get_questions_difficulties(qa_history)
    logging.info(f"  {len(original_difficulties):,} questions")

    logging.info(f"splitting train/test ({1 - split_ratio:.0%}/{split_ratio:.0%})...")
    train_df, test_df = mi.split_train_test_data_on_group(
        qa_history,
        [ColumnMapping.student_id, ColumnMapping.estimate_question_id],
        ratio=split_ratio,
    )
    logging.info(f"  train: {len(train_df):,} rows, test: {len(test_df):,} rows")

    logging.info("running MLE estimation...")
    estimation_track, df_estimation = run_mle(
        train_df,
        granularity_col,
        infer_mastery=infer_mastery,
        infer_item=infer_item,
        tune_discrimination=tune_discrimination,
        n_iter=n_iter,
        tol=tol,
    )

    logging.info("saving plots...")
    mle_track_plot(
        estimation_track,
        file_name=os.path.join(result_folder, "inference_track.png"),
        display=show_graph,
    )
    estimation_histogram(
        df_estimation,
        file_name=os.path.join(result_folder, "inference_histogram.png"),
        display=show_graph,
    )

    logging.info("saving results...")
    trained_mastery, trained_difficulty = get_result(
        df_estimation,
        granularity_col,
        original_questions_dificulties=original_difficulties,
        file_path=result_folder,
        outfile_suffix=outfile_suffix,
        using_window_col=using_window_col,
    )

    logging.info("calculating training ROC...")
    auc_train = roc_plot(
        df_estimation,
        "training ROC",
        file_name=os.path.join(result_folder, "inference_roc_train.png"),
        display=show_graph,
    )
    logging.info(f"  training AUC: {auc_train:.4f}")

    logging.info("calculating test ROC...")
    test_df_estimated = calc_test_result(
        trained_mastery, trained_difficulty, test_df, granularity_col, using_window_col=using_window_col
    )
    auc_test = roc_plot(
        test_df_estimated,
        "testing ROC",
        file_name=os.path.join(result_folder, "inference_roc_test.png"),
        display=show_graph,
    )
    logging.info(f"  test AUC: {auc_test:.4f}")

    if is_benchmark:
        benchmark_gbm(train_df, test_df, display=show_graph)
    return


if __name__ == "__main__":
    config = ConfigParser(
        interpolation=ExtendedInterpolation(), default_section="common"
    )
    config.read("config.ini")
    logfile = config["common"].get("logfile", None)

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s:%(message)s",
        filename=logfile,
        level=logging.INFO,
    )
    logging.info("\n" + "-" * 15 + " Model Inference " + "-" * 15)

    run(config)
