import csv
from collections import OrderedDict
from pathlib import Path
from itertools import zip_longest, count
import argparse
import pandas as pd
import numpy as np
import re
import functools
import sys

from typing import List, Optional, Pattern

NUM_FRAMES_PER_SECOND: int = 30


def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    print("checking duplicates")
    # get duplicated rows
    # see https://thispointer.com/pandas-find-duplicate-rows-in-a-dataframe-based-on-all-or-selected-columns-using-dataframe-duplicated-in-python/
    duplicated_mask = df.duplicated(subset=["frame"], keep=False)
    cols_to_drop = [c for c in df.columns if c.lower()[:7] == "eye_lmk"] + [
        "face_id",
        "timestamp",
        "success",
    ]
    if not np.any(duplicated_mask):
        print("No Duplicates")
        df_features = df.drop(columns=cols_to_drop)
        return df_features
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
                # drop failed rows
                group = group.drop(failed_group.index)

            if group.shape[0] == 1:
                # if only one row left (no duplicates in this group), return
                continue

            # largest_confidence = group.sort_values(by='confidence', ascending=False)["confidence"].iloc[0]
            largest_confidence = group["confidence"].max()
            confidence_mask = group["confidence"] < largest_confidence
            if 0 < np.count_nonzero(confidence_mask) < group.shape[0]:
                low_confidence_group = group[~confidence_mask]

                drop_indices = drop_indices.append(low_confidence_group.index)
                # drop rows with low confidence
                group = group.drop(low_confidence_group.index)

            if group.shape[0] == 1:
                # if only one row left (no duplicates in this group), return
                continue

        df_features = df.drop(columns=cols_to_drop)

        # keep the row closest to previous row (the row before the duplicate group)
        group_diff = np.sqrt((df_features.iloc[prev_index] - group) ** 2).sum(axis=1)
        far_dist_indices = group_diff.sort_values(ascending=False).iloc[:-1].index
        drop_indices = drop_indices.append(far_dist_indices)

        print("num duplication deleted: {}".format(drop_indices.shape[0]))
        return df_features.drop(index=drop_indices)


def parse_csv(input_file: Path) -> pd.DataFrame:
    # TODO: check if inplace operation is better
    print("reading input from ", input_file)
    df = pd.read_csv(input_file, header=0)
    df = df.rename(columns=lambda x: x.strip())

    # prefix = input_file.stem + "_"

    # drop face_id when returned
    df2 = handle_duplicates(df)
    # normalize timestamp
    df2["timestamp"] = (df2["frame"] - 1) / NUM_FRAMES_PER_SECOND

    # rename headers
    # see https://cmdlinetips.com/2018/03/how-to-change-column-names-and-row-indexes-in-pandas/
    # return df.rename(
    #     columns=lambda x: prefix + x
    #     if ("frame" not in x) and ("timestamp" not in x)
    #     else x
    # )
    return df2


def main(args):
    parser = argparse.ArgumentParser(
        description="take raw openface and produce clean files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_path", help="where to find input csv file",
    )
    parser.add_argument("output_path", help="where to place the csv")
    args = parser.parse_args()

    df = parse_csv(args.input_path)
    df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
