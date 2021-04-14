
import matplotlib.pyplot as plt
import pandas as pd
from pipeline.modeling.data_utils import TransformDF
from pipeline.modeling.datasets import TimeSeriesDataset
from pipeline.common.function_utils import timeit
from pipeline.common.optimize_pandas import optimize
from pipeline.modeling.data_to_df import LoadDF
import math


class MakeTurnsDataset():
    def __init__(self, config_pth, classes, max_roll, keep_unwindowed, normalize, fdf_path, include_mult) -> None:
        self.classes = ["speaking"]
        self.config_pth = config_pth
        self.max_roll = max_roll
        self.keep_unwindowed = keep_unwindowed
        self.normalize = normalize
        self.fdf_path = fdf_path

        # self.class_list = [i for _, v in classes.items() for i in v]

        positions = ["right","left","center"]
        skip = [4,5,14,18,19]
        sessions = range(1,28)

        sdfs, cdfs, at_mdfs, ang_mdfs, hdfs = [], [], [], [], []

        for session in sessions:
            if session not in skip:
                for person in positions:

                    looking_at, speaker_labels, confidences, gaze_angles, headpose = get_individuals_dataframes(session, person, "pose", "superlarge")
                    if include_mult:
                        at_multipliers = get_gazed_at_multiplier(person, looking_at)
                        ang_multipliers = get_gaze_angle_multiplier(person, gaze_angles)
                    
                    sdfs.append(speaker_labels)
                    cdfs.append(confidences)
                    if include_mult:
                        at_mdfs.append(at_multipliers)
                        ang_mdfs.append(ang_multipliers)
                    hdfs.append(headpose)

        speaker_labels = pd.concat(sdfs, axis=0)
        confidences = pd.concat(cdfs, axis=0)
        if include_mult:
            at_multipliers = pd.concat(at_mdfs, axis=0)
            ang_multipliers = pd.concat(ang_mdfs, axis=0)
        headposees = pd.concat(hdfs, axis=0)

        if include_mult:
            self.df = pd.concat([speaker_labels, confidences, at_multipliers, ang_multipliers, headposees], axis=1)
        else:
            self.df = pd.concat([speaker_labels, confidences, headposees], axis=1)

        self.df.reset_index(inplace=True,drop=True)
        self.df.to_feather(fdf_path)

    @timeit
    def transform_dataset(self, trial, df, model_params, shuffle, window_config):
        print("\n\n*****Transforming Dataset*******")
        # tdf = TransformDF()
        # rolling_window_size = trial.suggest_int("r_win_size", 1, self.max_roll)
        # step_size = trial.suggest_int("step_size", 1, 6)

        # df = tdf.apply_rolling_window(
        #     df,
        #     rolling_window_size,
        #     self.keep_unwindowed,
        #     window_config,
        # )
        # df = tdf.sub_sample(df, step_size)
        # if self.normalize:
        #     df = tdf.normalize_dataset(df, self.classes)

        subsample_perc = trial.suggest_int("sub_sample_neg_perc", 50, 95)

        dataset = TimeSeriesDataset(
            df,
            labels=self.classes,
            shuffle=shuffle,
            subsample_perc=subsample_perc,
            # data_hash=FILE_HASH,
        )
        dataset.setup_dataset(window=model_params["window"])
        # model_params["rolling_window_size"] = rolling_window_size
        # model_params["step_size"] = step_size
        model_params["subsample_perc"] = subsample_perc
        model_params["num_features"] = dataset.df.shape[1]
        model_params["class_weights"] = dataset.weights
        return dataset, model_params


def get_gazed_at_multiplier(person_of_interest, looking_at, add_to_multiply=.5, base_multiple=1):
    """Creates a 'per-frame' multiplier based on the # of people looking at the person_of_interest

    frame_multiplier = base_multiple + (add_to_multiply * [# of people looking at that person])

    note: Doesn't include the robot 'looking' at someone.

    Args:
        person_of_interest (str): subject under consideration as active speaker
        looking_at (df): df with group members as the columns and who they are looking at in each row
        add_to_multiply (float, optional): % to add for each person looking at the poi. Defaults to .5.
        base_multiple (int, optional): multiplier if no one is looking at the poi. Defaults to 1.

    Returns:
        [df]: a mutliplier value for every frame
    """
    # Set baseline multiple
    looking_at["multiple_at"] = base_multiple
    for p in ["left", "right", "center"]:
        looking_at.loc[looking_at[p]==person_of_interest,"multiple_at"] += add_to_multiply
    return looking_at[["multiple_at"]].copy()

def get_gaze_angle_multiplier(person_of_interest, angles_df, lower_bound=1, upper_bound=2):
    """Creates a 'per-frame' multiplier based on the average delta between group members gaze
    and the head of the person of interest.

    The goal is to map the angles from 0 (directly at the person) to 75 (the outer edge of the fov)
    to a range of max [2] to min [0].

    deg_angle = mean_angle*180/pi
    max = 0 * m + b
    min = 75 * m + max
    b=max
    m=(min-max)/75

    Args:
        person_of_interest (str): subject under consideration as active speaker
        angles_df (df): df of every combination of gaze-to-head angles. Columns are labeled
                        with ['person->subject'] headers such that the column contains the angle
                        between person's gaze the vector from the person's head to the subject's head
        rad_thresh (float, optional): [description]. Defaults to .7.

    Returns:
        [type]: [description]
    """
    angles_df["multiple_ang"] = lower_bound
    columns_of_interest = [f"{p}->{person_of_interest}" for p in ["left", "right", "center"] if p != person_of_interest]
    m = (lower_bound-upper_bound)/75

    angles_df["multiple_ang"] = angles_df[columns_of_interest].mean(axis=1) * (180/math.pi) * m + upper_bound
    # print(lower_bound, upper_bound)
    # print(angles_df["multiple_ang"].min(),angles_df["multiple_ang"].max())
    assert angles_df["multiple_ang"].max() <= upper_bound, "Check your math"
    return angles_df[["multiple_ang"]].copy()

def get_individuals_dataframes(session, person, direction_type, size, net="perfectmatch"):
    # Note: all csv's have headers so positions should be irrelevant

    # csv with a column for each speaker label with text labels for who they are gazing at
    looking_at = pd.read_csv(f"/media/chris/M2/2-Processed_Data/Gaze-Data/{session}/{direction_type}_at_{size}_cyl.csv")
    
    # csv with a column for each permutation of looker and subject with angle in radians
    # e.g. "left->right" | "left->center" | "right->left" | etc.
    gaze_angles = pd.read_csv(f"/media/chris/M2/2-Processed_Data/Gaze-Data/{session}/{direction_type}_ang.csv")

    # csv with a column for each speaker label with binary values for talking or not talking
    turns = pd.read_csv(f"/media/chris/M2/2-Processed_Data/Annotation-Turns/{session}/turns.csv")

    # csv with a single columns labeled "Confidence" and values from syncnet output
    confidences = pd.read_csv(f"/media/chris/M2/2-Processed_Data/{net}_output/pyavi/{session}{person[0]}/framewise_confidences.csv")

    headpose = pd.read_csv(f"/media/chris/M2/2-Processed_Data/Video-OpenFace/{session}/{person}.csv", usecols=["pose_Rx","pose_Ry"])
    
    # Sampled from 30 to 25 fps
    looking_at = looking_at[looking_at.index % 6 != 0].reset_index(drop=True)
    gaze_angles = gaze_angles[gaze_angles.index % 6 != 0].reset_index(drop=True)
    turns = turns[turns.index % 6 != 0].reset_index(drop=True)
    headpose = headpose[headpose.index % 6 != 0].reset_index(drop=True)

    # Get individual speaker from turns dataframe
    speaker_labels = turns[person].to_frame(name="speaking").reset_index(drop=True)
    
    
    max_len = min(speaker_labels.shape[0], confidences.shape[0], looking_at.shape[0], gaze_angles.shape[0])

    speaker_labels = speaker_labels.iloc[:max_len].fillna(0).reset_index(drop=True)
    confidences = confidences.iloc[:max_len].fillna(0).reset_index(drop=True)
    looking_at = looking_at.iloc[:max_len].fillna(0).reset_index(drop=True)
    gaze_angles = gaze_angles.iloc[:max_len].fillna(0).reset_index(drop=True)
    headpose = headpose.iloc[:max_len].fillna(0).reset_index(drop=True)

    return looking_at, speaker_labels, confidences, gaze_angles, headpose


if __name__ == '__main__':
    mds = MakeTurnsDataset('config_pth', 'classes', 'max_roll', 'keep_unwindowed', 'normalize', 'fdf_path')