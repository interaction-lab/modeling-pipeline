from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import sys

NUM_FRAMES_PER_SECOND: int = 30


def drop_columns(df: pd.DataFrame, drop_key: str, drop_list: list = ["face_id", "timestamp", "success"]):
    """Drop columns according to key or custom drop list

    Args:
        df (pd.DataFrame): [description]
        drop_key (str): [description]
        drop_list (list, optional): [description]. Defaults to ["face_id", "timestamp", "success"].

    Returns:
        [type]: clean df without columns
    """
    cols_to_drop = [c for c in df.columns if drop_key in c.lower()] + drop_list
    df = df.drop(columns=cols_to_drop)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from an Openface CSV

    TODO: double check this function is correct (is partially redundant?)

    Args:
        df (pd.DataFrame): dataframe to clean

    Returns:
        pd.DataFrame: cleaned dataframe
    """
    # see https://thispointer.com/pandas-find-duplicate-rows-in-a-dataframe-based-on-all-or-selected-columns-using-dataframe-duplicated-in-python/
    duplicated_mask = df.duplicated(subset=["frame"], keep=False)
    if not np.any(duplicated_mask):
        return df
    else:
        print("duplicate #: {}".format(np.count_nonzero(duplicated_mask)))
        df_frame = df[duplicated_mask].groupby("frame")
        drop_indices = pd.Index([])
        for key in df_frame.groups:
            group = df_frame.get_group(key)
            prev_index = group.index.min() - 1

            success_mask = group["success"] == 1
            if 0 < np.count_nonzero(success_mask) < group.shape[0]:
                failed_group = group[~success_mask]

                drop_indices = drop_indices.append(failed_group.index)
                group = group.drop(failed_group.index)

            if group.shape[0] == 1:
                continue

            # largest_confidence = group.sort_values(by='confidence', ascending=False)["confidence"].iloc[0]
            largest_confidence = group["confidence"].max()
            confidence_mask = group["confidence"] < largest_confidence
            if 0 < np.count_nonzero(confidence_mask) < group.shape[0]:
                low_confidence_group = group[~confidence_mask]

                drop_indices = drop_indices.append(low_confidence_group.index)
                group = group.drop(low_confidence_group.index)

            if group.shape[0] == 1:
                continue

        # keep the row closest to previous row (the row before the duplicate group)
        group_diff = np.sqrt((df.iloc[prev_index] - group) ** 2).sum(axis=1)
        far_dist_indices = group_diff.sort_values(ascending=False).iloc[:-1].index
        drop_indices = drop_indices.append(far_dist_indices)

        print("num duplication deleted: {}".format(drop_indices.shape[0]))
        return df.drop(index=drop_indices)


def reformat_csv(src_path: Path, dst_path: Path) -> pd.DataFrame:
    """Drops unused columns and removes duplicate rows

    This function is intended for cleaning up an openface csv with a single
    person detected.

    Args:
        src_path (Path): path to openface csv
        dst_path (Path): path to openface csv

    Returns:
        pd.DataFrame: [description]
    """
    df = pd.read_csv(src_path, header=0)
    df = df.rename(columns=lambda x: x.strip())

    df = drop_columns(df, "eye_lmk", drop_list=["face_id", "timestamp", "success"])
    df = remove_duplicates(df)

    # normalize timestamp
    df["timestamp"] = (df["frame"] - 1) / NUM_FRAMES_PER_SECOND

    df.to_csv(dst_path, index=False)
    return


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="take raw openface and produce clean files (for one person)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_path", help="where to find input csv file",
    )
    parser.add_argument("output_path", help="where to place the csv")
    args = parser.parse_args()
    return args


def main(args):
    args = parse_args(args)
    reformat_csv(args.input_path, args.output_path)


if __name__ == "__main__":
    main(sys.argv[1:])
