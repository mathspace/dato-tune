import logging
import os
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path
from typing import Dict, TextIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve

import model_inference as mi
from load_data import DataLoader
from utils import ColumnMapping


def load_inference_data(data_loader: DataLoader):
    # qa_history = data_loader.qa_history(add_default_values=False)
    # teacher_difficulty = data_loader.teacher_difficulty()
    # qa_history = qa_history.merge(teacher_difficulty[[ColumnMapping.question_id, "difficulty", "discrimination"]],
    #                               on=ColumnMapping.question_id, how="left", validate="m:1")

    df = data_loader.load_latern_responses()
    return df


def run_mle(
    train_data, granularity, infer_mastery=True, infer_item=True, n_iter=30, tol=0.1
):
    df = train_data.copy()
    # add default value as initial value of optimisation
    for col, default_value in zip(
        [ColumnMapping.mastery, ColumnMapping.difficulty, ColumnMapping.discrimination],
        [0.0, 0.0, 1.0],
    ):
        if col not in df.columns:
            df[col] = default_value
    df = df[
        [
            ColumnMapping.student_id,
            granularity,
            ColumnMapping.question_id,
            ColumnMapping.score,
            ColumnMapping.difficulty,
            ColumnMapping.discrimination,
            ColumnMapping.mastery,
        ]
    ]

    likelihood = mi.total_likelihood(df)
    estimation_tracking = [(0, "item", likelihood), (0, "mastery", likelihood)]
    for it in range(n_iter):
        if infer_item:
            df = mi.batch_item_estimation(df)
            likelihood = mi.total_likelihood(df)
            estimation_tracking.append((it + 1, "item", likelihood))
            logging.info(
                f"iteration: {it}, step: item estimation, total likelihood{likelihood}"
            )

        if infer_mastery:
            df = mi.batch_mastery_estimation(df, granularity_col=granularity)
            likelihood = mi.total_likelihood(df)
            estimation_tracking.append((it + 1, "mastery", likelihood))
            logging.info(
                f"iteration: {it}, step: mastery estimation, total likelihood{likelihood}"
            )

        if len(estimation_tracking) >= 4:
            benefit_mastery = estimation_tracking[-1][2] - estimation_tracking[-3][2]
            benefit_item = estimation_tracking[-2][2] - estimation_tracking[-4][2]
            if (benefit_mastery < tol) and (benefit_item < tol):
                logging.info(
                    f"optimisation stopped at iteration {it}, as likelihood improvement is less than the tolerance level {tol}"
                )
                break

    df[ColumnMapping.p_correct] = mi.p_correct(
        df[ColumnMapping.mastery].values,
        df[ColumnMapping.difficulty].values,
        df[ColumnMapping.discrimination].values,
    )
    estimation_tracking = pd.DataFrame.from_records(
        estimation_tracking, columns=["iter", "step", "likelihood"]
    )
    return estimation_tracking, df


def mle_track_plot(
    tracking, title="Likelihood per iteration", file_name=None, display=True
):
    p = sns.lineplot(x="iter", y="likelihood", hue="step", data=tracking)
    p.set_title(title)
    if file_name:
        plt.savefig(file_name)
        logging.info(f"mle_track_plot: {title} saved as {file_name}")
    if display:
        plt.show()
    return


def roc_plot(df, title="ROC curve", file_name=None, display=True):
    fpr, tpr, _ = roc_curve(df[ColumnMapping.score], df[ColumnMapping.p_correct])
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


def estimation_histogram(df, title="Histogram", file_name=None, display=True):
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(18, 5))
    df[ColumnMapping.difficulty].hist(bins=30, ax=ax[0])
    ax[0].set_title("Estimated difficulty")

    df[ColumnMapping.discrimination].hist(bins=30, ax=ax[1])
    ax[1].set_title("Estimated discrimination index")

    df[ColumnMapping.mastery].hist(bins=30)
    ax[2].set_title("Estimated mastery")

    plt.suptitle(title, fontsize=14)
    if file_name:
        plt.savefig(file_name)
        logging.info(f"estimation_histogram: {title} saved as {file_name}")
    if display:
        plt.show()
    return


def get_result(
    train_result,
    granularity,
    original_questions_dificulties: Dict[str, float],
    file_path=None,
    outfile: TextIO | None = None,
):
    estimated_mastery = (
        train_result.groupby([ColumnMapping.student_id, granularity], as_index=False)
        .agg(
            {
                ColumnMapping.mastery: "mean",
                ColumnMapping.question_id: pd.Series.nunique,
            }
        )
        .rename(columns={ColumnMapping.question_id: "n_question"})
    )
    estimated_difficulty = (
        train_result.groupby([ColumnMapping.question_id], as_index=False)
        .agg(
            {
                ColumnMapping.difficulty: "mean",
                ColumnMapping.discrimination: "mean",
                ColumnMapping.student_id: pd.Series.nunique,
            }
        )
        .rename(columns={ColumnMapping.student_id: "n_student"})
    )

    estimated_difficulty["OriginalDifficulty"] = estimated_difficulty[
        ColumnMapping.question_id
    ].apply(lambda q_id: original_questions_dificulties.get(q_id))
    estimated_difficulty["CalibratedDifficulty"] = (
        estimated_difficulty[ColumnMapping.difficulty]
        - estimated_difficulty[ColumnMapping.difficulty].min()
    ) / (
        estimated_difficulty[ColumnMapping.difficulty].max()
        - estimated_difficulty[ColumnMapping.difficulty].min()
    )
    estimated_difficulty["DifficultiesDifference abs(Original-Calibrated)"] = (
        estimated_difficulty["CalibratedDifficulty"]
        - estimated_difficulty["OriginalDifficulty"]
    ).abs()

    if outfile:
        estimated_difficulty.to_csv(outfile, index=False)
        logging.info("estimated difficulty saved to outfile")

    if file_path:
        mastery_file = os.path.join(file_path, "estimated_mastery.csv")
        difficulty_file = os.path.join(file_path, "estimated_item.csv")
        estimated_mastery.to_csv(mastery_file)
        estimated_difficulty.to_csv(difficulty_file, index=False)
        logging.info(
            f"estimated mastery and difficulty are saved as files {mastery_file} and {difficulty_file}"
        )

    return estimated_mastery, estimated_difficulty


def calc_test_result(estimated_mastery, estimated_difficulty, test_data, granularity):
    # predict on the test data
    df = (
        test_data.drop(
            [ColumnMapping.difficulty, ColumnMapping.discrimination],
            axis=1,
            errors="ignore",
        )
        .merge(
            estimated_mastery, on=[ColumnMapping.student_id, granularity], how="inner"
        )
        .merge(estimated_difficulty, on=[ColumnMapping.question_id], how="inner")
    )
    df[ColumnMapping.p_correct] = mi.p_correct(
        df[ColumnMapping.mastery].values,
        df[ColumnMapping.difficulty].values,
        df[ColumnMapping.discrimination].values,
    )
    logging.info(
        f"{np.round(df.shape[0] / test_data.shape[0], 2)} of test data has estimated mastery and difficulty"
    )
    return df


def benchmark_gbm(train_data, test_data, **kwargs):
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import GridSearchCV

    show_graph = kwargs.get("display", kwargs.get("show_graph", False))
    train_data["question_num_id"] = (
        train_data[ColumnMapping.question_id].astype("category").cat.codes
    )
    test_df_sub = test_data.merge(
        train_data[[ColumnMapping.student_id]].drop_duplicates(),
        on=[ColumnMapping.student_id],
    ).merge(
        train_data[[ColumnMapping.question_id, "question_num_id"]].drop_duplicates(),
        on=[ColumnMapping.question_id],
    )
    X_train = train_data[[ColumnMapping.student_id, "question_num_id"]].values
    y_train = train_data[ColumnMapping.score].values

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
        pd.DataFrame({ColumnMapping.p_correct: proba, ColumnMapping.score: y_train}),
        "GBM training ROC",
        file_name=kwargs.get("train_roc_file", None),
        display=show_graph,
    )
    logging.info(f"Benchmark: training ROC AUC score is {auc_train}")

    X_test = test_df_sub[[ColumnMapping.student_id, "question_num_id"]].values
    y_test = test_df_sub[ColumnMapping.score].values
    proba = model.predict_proba(X_test)[:, 1]
    auc_test = roc_plot(
        pd.DataFrame({ColumnMapping.p_correct: proba, ColumnMapping.score: y_test}),
        "GBM testing ROC",
        file_name=kwargs.get("train_roc_file", None),
        display=show_graph,
    )
    logging.info(f"Benchmark: testing ROC AUC score is {auc_test}")

    return model


def get_questions_difficulties(df) -> Dict[str, float]:
    return {
        question_id: difficulty
        for question_id, difficulty in df[
            [ColumnMapping.question_id, ColumnMapping.difficulty]
        ]
        .drop_duplicates()
        .values
    }


def run(config: ConfigParser, df: pd.DataFrame, outfile: TextIO):
    inference_config = config["inference"]

    result_folder = Path(inference_config["result_folder"])
    result_folder.mkdir(exist_ok=True, parents=True)
    granularity_col = inference_config["granularity_col"]
    n_iter = inference_config.getint("n_iter", 15)
    tol = inference_config.getfloat("tol", 0.1)
    infer_mastery = inference_config.getboolean("infer_mastery", True)
    infer_item = inference_config.getboolean("infer_item", True)
    is_benchmark = inference_config.getboolean("is_benchmark", False)
    show_graph = inference_config.getboolean("show_graph", False)
    random_seed = inference_config.getint("random_seed", 123)
    min_obs = inference_config.getint("min_obs", 10)
    split_ratio = inference_config.getfloat("split_ratio", 0.2)

    np.random.seed(random_seed)

    qa_history = df
    qa_history = mi.remove_groups_with_insufficient_data(
        qa_history, [ColumnMapping.question_id], min_obs
    )
    original_difficulties = get_questions_difficulties(qa_history)
    train_df, test_df = mi.split_train_test_data_on_group(
        qa_history,
        [ColumnMapping.student_id, ColumnMapping.question_id],
        ratio=split_ratio,
    )
    logging.info(
        f"train dataset has {train_df.shape[0]} observations; test dataset has {test_df.shape[0]} observations"
    )

    estimation_track, df_estimation = run_mle(
        train_df,
        granularity_col,
        infer_mastery=infer_mastery,
        infer_item=infer_item,
        n_iter=n_iter,
        tol=tol,
    )
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

    trained_mastery, trained_difficulty = get_result(
        df_estimation,
        granularity_col,
        original_questions_dificulties=original_difficulties,
        file_path=result_folder,
        outfile=outfile,
    )
    auc_train = roc_plot(
        df_estimation,
        "training ROC",
        file_name=os.path.join(result_folder, "inference_roc_train.png"),
        display=show_graph,
    )
    logging.info(f"training ROC AUC score is {auc_train}")

    test_df_estimated = calc_test_result(
        trained_mastery, trained_difficulty, test_df, granularity_col
    )
    auc_test = roc_plot(
        test_df_estimated,
        "testing ROC",
        file_name=os.path.join(result_folder, "inference_roc_test.png"),
        display=show_graph,
    )
    logging.info(f"testing ROC AUC score is {auc_test}")

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
