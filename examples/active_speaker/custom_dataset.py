
import matplotlib.pyplot as plt
import pandas as pd
from pipeline.modeling.data_utils import TransformDF
from pipeline.modeling.datasets import TimeSeriesDataset
from pipeline.common.function_utils import timeit
from pipeline.common.optimize_pandas import optimize
from pipeline.modeling.data_to_df import LoadDF
import math


class MakeTurnsDataset():
    def __init__(self, train_sessions, val_sessions, fdf_path, features=["at","ang","head","perfectmatch"]) -> None:
        self.fdf_path = fdf_path

        df = self.get_df(train_sessions,"chris", features)
        df.to_feather(fdf_path+".train")

        df = self.get_df(val_sessions,"chris", features)
        df.to_feather(fdf_path+".val")

        df = self.get_df(range(1,16),"kalin", features)
        df.to_feather(fdf_path+".test")

    def get_df(self,sessions,dataset, features):
        if dataset == "chris":
            skip = [4,5,14,18,19]
        else:
            skip=[]
        positions = ["right","left","center"]
        sdfs, pcdfs, scdfs, at_mdfs, ang_mdfs = [], [], [], [], []
        for session in sessions:
            if session not in skip:
                for person in positions:

                    looking_at, speaker_labels, sync, perf, gaze_angles = get_individuals_dataframes(session, person, "pose", "superlarge", dataset)
                    at_multipliers = get_gazed_at_multiplier(person, looking_at)
                    ang_multipliers = get_gaze_angle_multiplier(person, gaze_angles)
                    
                    sdfs.append(speaker_labels)
                    pcdfs.append(perf)
                    scdfs.append(sync)
                    at_mdfs.append(at_multipliers)
                    ang_mdfs.append(ang_multipliers)
        dfs = {}
        speaker_labels = pd.concat(sdfs, axis=0)
        dfs["perfectmatch"] = pd.concat(pcdfs, axis=0)
        dfs["syncnet"] = pd.concat(scdfs, axis=0)
        dfs["at"] = pd.concat(at_mdfs, axis=0)
        dfs["ang"] = pd.concat(ang_mdfs, axis=0)

        to_concat = [speaker_labels]

        for f in features:
            to_concat.append(dfs[f])

        df = pd.concat(to_concat, axis=1)
        
        df.reset_index(inplace=True,drop=True)
        return df




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
    
def get_individuals_dataframes(session, person, direction_type, size, dataset):
    # Note: all csv's have headers so positions should be irrelevant

    if dataset=="kalin":
        base_path = "/home/chris/code/modeling-pipeline/data/active_speaker/"
        # csv with a column for each speaker label with text labels for who they are gazing at
        looking_at = pd.read_csv(f"{base_path}/kinect_pose/{session}G3_KINECT_DISCRETE_{size.upper()}.csv")

        # csv with a column for each permutation of looker and subject with angle in radians
        # e.g. "left->right" | "left->center" | "right->left" | etc.
        gaze_angles = pd.read_csv(f"{base_path}/kinect_pose/{session}G3_KINECT_CONTINUOUS.csv")

        # csv with a column for each speaker label with binary values for talking or not talking
        turns = pd.read_csv(f"{base_path}/kinect_pose/{session}G3_VAD.csv")

        # csv with a single columns labeled "Confidence" and values from syncnet output
        if person == 'center':
            sconfidences = pd.read_csv(f"{base_path}/kinect_pose/{session}G3C_SYNCNET.csv")
            pconfidences = pd.read_csv(f"{base_path}/kinect_pose/{session}G3C_PERFECTMATCH.csv")
        if person == 'left':
            sconfidences = pd.read_csv(f"{base_path}/kinect_pose/{session}G3L_SYNCNET.csv")
            pconfidences = pd.read_csv(f"{base_path}/kinect_pose/{session}G3L_PERFECTMATCH.csv")
        if person == 'right':
            sconfidences = pd.read_csv(f"{base_path}/kinect_pose/{session}G3R_SYNCNET.csv")
            pconfidences = pd.read_csv(f"{base_path}/kinect_pose/{session}G3R_PERFECTMATCH.csv")

    if dataset=="chris":
        # Note: all csv's have headers so positions should be irrelevant
        base_path = f"/home/chris/code/modeling-pipeline/data/active_speaker/facilitator"

        # csv with a column for each speaker label with text labels for who they are gazing at
        looking_at = pd.read_csv(f"{base_path}/Gaze-Data/{session}/{direction_type}_at_{size}_cyl.csv")
        
        # csv with a column for each permutation of looker and subject with angle in radians
        # e.g. "left->right" | "left->center" | "right->left" | etc.
        gaze_angles = pd.read_csv(f"{base_path}/Gaze-Data/{session}/{direction_type}_ang.csv")

        # csv with a column for each speaker label with binary values for talking or not talking
        turns = pd.read_csv(f"{base_path}/Annotation-Turns/{session}/turns.csv")

        # csv with a single columns labeled "Confidence" and values from syncnet output
        sconfidences = pd.read_csv(f"{base_path}/syncnet_confidences/pyavi/{session}{person[0]}/framewise_confidences.csv")
        pconfidences = pd.read_csv(f"{base_path}/perfectmatch_confidences/pyavi/{session}{person[0]}/framewise_confidences.csv")

    # Sampled from 30 to 25 fps
    looking_at = looking_at[looking_at.index % 6 != 0].reset_index(drop=True)
    gaze_angles = gaze_angles[gaze_angles.index % 6 != 0].reset_index(drop=True)
    turns = turns[turns.index % 6 != 0].reset_index(drop=True)

    # Get individual speaker from turns dataframe
    speaker_labels = turns[person].to_frame(name="speaking").reset_index(drop=True)

    max_len = min(speaker_labels.shape[0], pconfidences.shape[0], sconfidences.shape[0], looking_at.shape[0], gaze_angles.shape[0])

    sconfidences.rename(columns = {'Confidence' : 'sConfidence'}, inplace = True)
    pconfidences.rename(columns = {'Confidence' : 'pConfidence'}, inplace = True)

    speaker_labels = speaker_labels.iloc[:max_len].fillna(0).reset_index(drop=True)
    pconfidences = pconfidences.iloc[:max_len].fillna(0).reset_index(drop=True)
    sconfidences = sconfidences.iloc[:max_len].fillna(0).reset_index(drop=True)
    looking_at = looking_at.iloc[:max_len].fillna(0).reset_index(drop=True)
    gaze_angles = gaze_angles.iloc[:max_len].fillna(0).reset_index(drop=True)

    # return looking_at, speaker_labels, sync_confidences, perf_confidences, gaze_angles, headpose
    return looking_at, speaker_labels, sconfidences, pconfidences, gaze_angles

# if __name__ == '__main__':
    # mds = MakeTurnsDataset('config_pth', 'classes', 'max_roll',',', 'fdf_path', 'syncnet')