import logging
import os
import warnings
from configparser import ConfigParser

import git


class ColumnMapping:
    """
    STUDENT_GRADE_ID
    SCHOOL_ID
    ID
    USER_UUID
    USER_ID
    CHECKIN_ID
    QUESTION_VERSION_ID
    OUTCOME_CODE
    COLD_START_DIFFICULTY
    SKILL_ID
    OUTCOME_ID
    CURRICULUM_ID
    GRADE_SUBSTRAND_ID
    GRADE_STRAND_ID
    GRADE_ID
    STRAND_ID
    DURATION
    RECOMMENDATION_REASON_KEYWORD
    CREATED_AT
    WINDOW_INDEX"""

    question_id = "QUESTION_PUBLIC_ID"
    student_id = "STUDENT_ID"
    grade_strand_id = "GRADE_STRAND_ID"
    grade_substrand_id = "GRADE_SUBSTRAND_ID"
    skill_id = "SKILL_ID"
    difficulty = "COLD_START_DIFFICULTY"
    result = "RESULT"
    score = "SCORE"
    dummy = "dummy"
    completed_at = "CREATED_AT"
    curriculum_id = "CURRICULUM_ID"
    discrimination = "discrimination"
    mastery = "mastery"
    p_correct = "p_correct"
    window_index = "WINDOW_INDEX"


def get_git_info():
    repo = git.Repo(search_parent_directories=True)
    branch = repo.active_branch
    hash = repo.head.object.hexsha
    return branch, hash


def config_to_string(config: ConfigParser):
    config_string = ""
    for ss in config.sections():
        config_string += f"\n[{ss}]"
        for key, value in config[ss].items():
            config_string += f"\n{key}: {value}"
        config_string += "\n"
    return config_string


def delete_result(config: ConfigParser):
    result_folder = config["common"]["result_folder"]
    if not os.path.exists(result_folder):
        logging.warning(
            f"result folder {result_folder} does not exist! Creating the folder and exiting result deletion !"
        )
        os.mkdir(result_folder)
        return False
    logging.warning(f"removing all files in {result_folder} !")
    files = os.listdir(result_folder)
    for file in files:
        try:
            os.remove(os.path.join(result_folder, file))
        except PermissionError:
            logging.warning(f"{file} cannot be deleted !")
    return True


def delete_logfile(config: ConfigParser):
    logfile = config["common"]["logfile"]
    if not os.path.exists(logfile):
        warnings.warn(f"logfile {logfile} does not exist, creating a new one instead !")
    else:
        os.remove(logfile)
    f = open(logfile, "w+")
    f.close()
    return
