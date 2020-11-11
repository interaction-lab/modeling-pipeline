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
    # get duplicated rows
    # see https://thispointer.com/pandas-find-duplicate-rows-in-a-dataframe-based-on-all-or-selected-columns-using-dataframe-duplicated-in-python/
    duplicated_mask = df.duplicated(subset=["frame"], keep=False)
    if not np.any(duplicated_mask):
        return df

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

        cols_to_drop = [c for c in df.columns if c.lower()[:7] == "eye_lmk"] + [
            "frame",
            "face_id",
            "timestamp",
            "success",
        ]

        df_features = df.drop(columns=cols_to_drop)

        # keep the row closest to previous row (the row before the duplicate group)
        group_diff = np.sqrt((df_features.iloc[prev_index] - group) ** 2).sum(axis=1)
        far_dist_indices = group_diff.sort_values(ascending=False).iloc[:-1].index
        drop_indices = drop_indices.append(far_dist_indices)

    print("num duplication deleted: {}".format(drop_indices.shape[0]))
    return df.drop(index=drop_indices)


def interpolate_missing_feature(
    df: pd.DataFrame, method: str = "spline", order: int = 2
) -> pd.DataFrame:
    non_feature_columns = [
        col for col in df.columns if "confidence" in col or "success" in col
    ]
    non_feature_columns += ["frame", "timestamp"]

    df_features = df.drop(columns=non_feature_columns)
    df_output: pd.DataFrame = df.copy()

    num_empty_feature = np.count_nonzero(df_features.isna())
    print(
        "feature sparsity before interpolation: {:%} ({} / {})".format(
            num_empty_feature / df_features.size, num_empty_feature, df_features.size
        )
    )

    # see https://www.geeksforgeeks.org/python-pandas-dataframe-interpolate/
    # interpolate missing features
    df_new_features = df_features.interpolate(
        method=method, order=order, limit_direction="both"
    )

    num_empty_feature = np.count_nonzero(df_new_features.isna())
    if num_empty_feature > 0:
        print(
            "Error: expecting 0 missing feature but get {}".format(num_empty_feature),
            file=sys.stderr,
            flush=True,
        )
    print(
        "feature sparsity after interpolation: {:%} ({} / {})".format(
            num_empty_feature / df_new_features.size,
            num_empty_feature,
            df_new_features.size,
        )
    )

    # replace original features with interpolated features
    df_output[df_output.columns.difference(non_feature_columns)] = df_new_features

    return df_output


def parse_csv(input_file: Path) -> pd.DataFrame:
    # TODO: check if inplace operation is better
    df = pd.read_csv(input_file, header=0)
    df = df.rename(columns=lambda x: x.strip())

    prefix = input_file.stem + "_"

    # drop face_id when returned
    df = handle_duplicates(df).drop(columns=["face_id"])
    # normalize timestamp
    df["timestamp"] = (df["frame"] - 1) / NUM_FRAMES_PER_SECOND

    # rename headers
    # see https://cmdlinetips.com/2018/03/how-to-change-column-names-and-row-indexes-in-pandas/
    return df.rename(
        columns=lambda x: prefix + x
        if ("frame" not in x) and ("timestamp" not in x)
        else x
    )


def combine_csv(
    input_dir: Path,
    input_files: List[str],
    extension_filter: Pattern,
    interpolation_method: str = "linear",
) -> Optional[pd.DataFrame]:
    input_data_frames = list()

    for filename in input_files:
        file_item = input_dir / filename

        # skip if not all input files in this folder is present
        if not file_item.is_file() or not extension_filter.search(file_item.suffix):
            return None

        input_data_frames.append(parse_csv(file_item))

    # create a DataFrame for frame number and timestamp normalization
    max_frame_number = max([int(df["frame"].max()) for df in input_data_frames])
    frame_range = np.arange(0, max_frame_number) + 1
    frame_df = pd.DataFrame(
        data={
            "frame": frame_range,
            "timestamp": (frame_range - 1) / NUM_FRAMES_PER_SECOND,
        }
    )

    # https://stackoverflow.com/questions/23668427/pandas-three-way-joining-multiple-dataframes-on-columns/23671390
    # outer join all data to produce one table
    output_data = functools.reduce(
        lambda df1, df2: df1.merge(
            df2, on=["frame", "timestamp"], how="outer", sort=True
        ),
        input_data_frames,
    )

    # fill in empty value to skipped frames
    output_data = pd.merge(
        output_data, frame_df, on=["frame", "timestamp"], how="outer", sort=True
    )

    # interpolate empty frames
    # return interpolate_missing_feature(output_data, method=interpolation_method)
    return output_data


def process_files(
    input_dir: str,
    output_dir: str,
    input_files: List[str],
    base_output_filename: str,
    interpolation_method: str,
) -> None:
    if len(input_files) == 0:
        return

    input_dir_obj = Path(input_dir)
    output_dir_obj = Path(output_dir)

    output_dir_obj.mkdir(parents=True, exist_ok=True)

    # regex string for head pose log file extension
    extension_filter = re.compile(r"\.(csv)")

    for dir_item in input_dir_obj.iterdir():
        if not dir_item.is_dir():
            continue

        # get output filename
        output_filename = "study_" + dir_item.stem + "_" + base_output_filename
        print("processing {}...".format(output_filename))

        output_data = combine_csv(
            dir_item,
            input_files,
            interpolation_method=interpolation_method,
            extension_filter=extension_filter,
        )
        if output_data is None:
            print(
                "Error when parsing files for {}. Abort".format(output_filename),
                file=sys.stderr,
                flush=True,
            )
            continue

        # save to output
        print("saving {}...".format(output_filename))
        output_data.to_csv(output_dir_obj / output_filename, index=False)
        print(output_filename + " saved!\n")


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine input csv files into one output file"
    )

    parser.add_argument(
        "--input_filenames",
        help="names of input csv files",
        dest="input_filenames",
        type=str,
        nargs="+",
        required=False,
        default=["center.csv", "left.csv", "right.csv"],
    )

    parser.add_argument(
        "--output_filename",
        help="names of the output csv file",
        dest="output_filename",
        type=str,
        required=False,
        default="feature.csv",
    )

    parser.add_argument(
        "--interpolation_method",
        help="method for missing feature interpolation",
        dest="interpolation_method",
        type=str,
        required=False,
        default="linear",
    )

    parser.add_argument(
        "-i",
        "--input_dir",
        help="directory of the input csv files",
        dest="input_dir",
        type=str,
        required=False,
        default="Data/input",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        help="directory for the output csv files",
        dest="output_dir",
        type=str,
        required=False,
        default="Data/output",
    )

    return parser


if __name__ == "__main__":
    parsedArgs = get_arg_parser().parse_args()

    process_files(
        parsedArgs.input_dir,
        parsedArgs.output_dir,
        parsedArgs.input_filenames,
        parsedArgs.output_filename,
        parsedArgs.interpolation_method,
    )
