import pandas as pd
import numpy as np
from collections import Counter

of_cols = [
    "of_AU01_c",
    "of_AU02_c",
    "of_AU04_c",
    "of_AU05_c",
    "of_AU06_c",
    "of_AU07_c",
    "of_AU09_c",
    "of_AU10_c",
    "of_AU12_c",
    "of_AU14_c",
    "of_AU15_c",
    "of_AU17_c",
    "of_AU20_c",
    "of_AU23_c",
    "of_AU25_c",
    "of_AU26_c",
    "of_AU28_c",
    "of_AU45_c",
    "of_gaze_0_x",
    "of_gaze_0_y",
    "of_gaze_0_z",
    "of_gaze_1_x",
    "of_gaze_1_y",
    "of_gaze_1_z",
    "of_gaze_angle_x",
    "of_gaze_angle_y",
    "of_gaze_distance",
    "of_gaze_distance_x",
    "of_gaze_distance_y",
    "of_pose_Rx",
    "of_pose_Ry",
    "of_pose_Rz",
    "of_pose_Tx",
    "of_pose_Ty",
    "of_pose_Tz",
    "of_pose_distance",
]

# round_down features: features that are integers, if the median is not an integer round down
int_features = [
    "engagement",
    "C_Valence",
    "C_Arousal",
    "C_Seat",
    "of_AU01_c",
    "of_AU02_c",
    "of_AU04_c",
    "of_AU05_c",
    "of_AU06_c",
    "of_AU07_c",
    "of_AU09_c",
    "of_AU10_c",
    "of_AU12_c",
    "of_AU14_c",
    "of_AU15_c",
    "of_AU17_c",
    "of_AU20_c",
    "of_AU23_c",
    "of_AU25_c",
    "of_AU26_c",
    "of_AU28_c",
    "of_AU45_c",
    "of_success",
    "op_num_people",
    "ros_diff_1",
    "ros_diff_2",
    "ros_diff_3",
    "ros_diff_4",
    "ros_diff_5",
    "ros_games_session",
    "ros_mistakes_game",
    "ros_mistakes_session",
    "ros_in_game",
    "ros_skill_EM",
    "ros_skill_NC",
    "ros_skill_OS",
]

mode_cols = [
    "ros_activity",
    "ros_difficulty",
    "ros_skill",
    "ros_GAME_STATE",
    "ros_PARTICIPANT_STATE",
    "ros_ROBOT_STATE",
]

contains_cols = [
    "ros_game_correct",
    "ros_game_incorrect",
    "ros_game_start",
    "ros_mistake_made",
]

participants = [5, 7, 9, 17]

for p in participants:
    file = "p" + str(p) + "_openpose_master_seat.csv"
    data = pd.read_csv(file)

    data = data.sort_values(["session_num", "timestamp"], ascending=[True, True])
    data = data.reset_index(drop=True)

    # Overwrite open face features with NaN in rows where not successful
    for f in of_cols:
        data.loc[data["of_success"] == 0, f] = np.nan

    # Make sure all values numeric
    for f in data.columns:
        if f not in mode_cols and f not in contains_cols:
            data[f] = pd.to_numeric(data[f], errors="coerce")

    # Create Feature Dictionary for Each Window
    curr = 0
    window_size = 29
    increment = 15

    checkpoint = 0
    check_increment = 10000
    print("Total", len(data))

    to_add = []
    while curr + window_size < len(data):
        if curr >= checkpoint:
            print(curr)
            checkpoint += check_increment

        # adjust window start if this window contains a session_change,
        # ignore end of sessions
        while (
            data.loc[curr + window_size, "session_num"] != data.loc[curr, "session_num"]
        ):
            curr += 1

        window = data.loc[curr : curr + window_size]
        append = {}  # feature dictionary for this window
        for i in data.columns:
            # Contains Cols = 1 iff exists a 1 in the window
            if i in contains_cols:
                append[i] = window[i].sum()
            # Mode Cols: take most common value in the window
            elif i in mode_cols:
                occurence_count = Counter(window[i])
                append[i] = occurence_count.most_common(1)[0][0]
            # All other cols: if NaN for majority of window = NaN
            elif (window[i].isna().sum()) >= ((window_size + 1) / 2):
                append[i] = np.nan
                if i in int_features:
                    append[i + "_change"] = np.nan
                else:
                    append[i + "_var"] = np.nan
            # Otherwise, if not NaN for majority, use median.
            else:
                append[i] = np.nanmedian(window[i])
                if i in int_features:
                    check = np.nanvar(window[i])
                    if check > 0:
                        append[i + "_change"] = 1
                    else:
                        append[i + "_change"] = 0
                else:
                    append[i + "_var"] = np.nanvar(window[i])
        to_add.append(append)
        curr += increment

    # Create DataFrame with Windows
    cols = []
    for i in data.columns:
        cols.append(i)
        if i in int_features:
            cols.append(i + "_change")
        elif i not in mode_cols and i not in contains_cols:
            cols.append(i + "_var")

    answer = pd.DataFrame(columns=cols)
    answer = answer.append(to_add, ignore_index=True, sort=True)

    # Adjust, round some features
    answer["timestamp"] = round(answer["timestamp"], 1)

    for i in int_features:
        answer[i] = np.floor(answer[i])

    answer = answer.drop(
        columns=["participant_var", "timestamp_var", "session_num_var"]
    )
    answer.to_csv(
        "p" + str(p) + "_windowed_openpose_before_segmentation_1s.csv", index=False
    )
