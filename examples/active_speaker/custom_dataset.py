
import matplotlib.pyplot as plt
import pandas as pd
from pipeline.modeling.data_utils import TransformDF
from pipeline.modeling.datasets import TimeSeriesDataset
from pipeline.common.function_utils import timeit
from pipeline.common.optimize_pandas import optimize
from pipeline.modeling.data_to_df import LoadDF



class MakeTurnsDataset():
    def __init__(self, config_pth, classes, max_roll, keep_unwindowed, normalize, fdf_path) -> None:
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

                    looking_at, speaker_labels, confidences, gaze_angles, headpose = get_support_group_dataframes(session, person, "pose", "superlarge")
                    at_multipliers = get_gazed_at_multiplier(person, looking_at)
                    ang_multipliers = get_gaze_angle_multiplier(person, gaze_angles)
                    
                    sdfs.append(speaker_labels)
                    cdfs.append(confidences)
                    at_mdfs.append(at_multipliers)
                    ang_mdfs.append(ang_multipliers)
                    hdfs.append(headpose)

        speaker_labels = pd.concat(sdfs, axis=0)
        confidences = pd.concat(cdfs, axis=0)
        at_multipliers = pd.concat(at_mdfs, axis=0)
        ang_multipliers = pd.concat(ang_mdfs, axis=0)
        headposees = pd.concat(hdfs, axis=0)
        self.df = pd.concat([speaker_labels, confidences, at_multipliers, ang_multipliers, headposees], axis=1)
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



def get_gazed_at_multiplier(person_of_interest, looking_at, add_to_multiply=.5, base_multiple=0):
    """ Each time a person is being gazed at the multiple is the number of people that they

    frame_multiplier = base_multiple + (add_to_multiply * # of people looking at that person)

    Doesn't include the robot 'looking' at someone.
    """
    # Set baseline multiple
    looking_at["at"] = base_multiple
    for p in ["left", "right", "center"]:
        looking_at.loc[looking_at[p]==person_of_interest,"at"] += add_to_multiply
    return looking_at[["at"]].copy()

def get_gaze_angle_multiplier(person_of_interest, angles_df, rad_thresh=.7):
    angles_df["ang"] = 1
    columns_of_interest = [f"{p}->{person_of_interest}" for p in ["left", "right", "center"] if p != person_of_interest]
    angles_df["ang"] += rad_thresh - angles_df[columns_of_interest].mean(axis=1) #.7=40
    # print(angles_df)
    return angles_df[["ang"]].copy()


def get_support_group_dataframes(session, person, direction_type, size):
    gaze_angles = pd.read_csv(f"/media/chris/M2/2-Processed_Data/Gaze-Data/{session}/{direction_type}_ang.csv")
    looking_at = pd.read_csv(f"/media/chris/M2/2-Processed_Data/Gaze-Data/{session}/{direction_type}_at_{size}_cyl.csv")
    # Sampled from 30 to 25 fps
    looking_at = looking_at[looking_at.index % 6 != 0].reset_index(drop=True)

    turns = pd.read_csv(f"/media/chris/M2/2-Processed_Data/Annotation-Turns/{session}/turns.csv")
    # Sampled from 30 to 25 fps
    turns = turns[turns.index % 6 != 0].reset_index(drop=True)
    speaker_labels = turns[person].to_frame(name="speaking").reset_index(drop=True)

    confidences = pd.read_csv(f"/media/chris/M2/2-Processed_Data/syncnet_output/pyavi/{session}{person[0]}/framewise_confidences.csv")
    headpose = pd.read_csv(f"/media/chris/M2/2-Processed_Data/Video-OpenFace/{session}/{person}.csv", usecols=["pose_Rx","pose_Ry"])
    headpose = headpose[headpose.index % 6 != 0].reset_index(drop=True)
    max_len = min(speaker_labels.shape[0], confidences.shape[0], looking_at.shape[0])

    speaker_labels = speaker_labels.iloc[:max_len].fillna(0).reset_index(drop=True)
    confidences = confidences.iloc[:max_len].fillna(0).reset_index(drop=True)
    looking_at = looking_at.iloc[:max_len].fillna(0).reset_index(drop=True)
    gaze_angles = gaze_angles.iloc[:max_len].fillna(0).reset_index(drop=True)
    headpose = headpose.iloc[:max_len].fillna(0).reset_index(drop=True)

    return looking_at, speaker_labels, confidences, gaze_angles, headpose


if __name__ == '__main__':
    mds = MakeTurnsDataset('config_pth', 'classes', 'max_roll', 'keep_unwindowed', 'normalize', 'fdf_path')