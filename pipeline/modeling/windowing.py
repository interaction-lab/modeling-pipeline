import pandas as pd
import numpy as np
from collections import Counter
import yaml
import argparse
import sys


class Windowing:
    """
    Basic wrapper on pandas for windowing and subsampling a dataset.

    The windowing class is intended to provide a convenient wrapper on pandas rolling
    functionality. There is currently a distinction between two types of features:
        - continous features: produce mean and variance across the window.
        - binary features: produce any of [median, mode, max]

    feature types are specified in [example]_config.yml which lists the headers of
    columns according to the type of data contained (continuous or binary).

    returns:
        windowed dataframe without the original features.

    """

    def __init__(self, csv, config="./windowing_config.yml"):
        """[summary]

        Args:
            csv ([type]): [description]
            config (str, optional): [description]. Defaults to "./windowing_config.yml".
        """
        with open(config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.df = pd.read_csv(csv)
        print(self.df)

    def window_dataframe(self, window_size=30, step=10, binary_mode="median"):
        # Contains Cols = 1 iff exists a 1 in the window
        # Mode Cols: take most common value in the window
        # All other cols: if NaN for majority of window = NaN
        # Otherwise, if not NaN for majority, use median.
        self.windowed_df = pd.DataFrame()

        if self.config["float_features"]:
            self.window_float(window_size)
        if self.config["binary_features"]:
            self.window_binary(window_size, mode=binary_mode)

        self.windowed_df = self.windowed_df.loc[
            self.windowed_df.index[np.arange(len(self.windowed_df)) % step == 1]
        ]
        print(self.windowed_df)
        return self.windowed_df

    def window_float(self, window_size):
        mean_cols = [f + "_mean" for f in self.config["float_features"]]
        var_cols = [f + "_var" for f in self.config["float_features"]]

        self.windowed_df[mean_cols] = (
            self.df[self.config["float_features"]]
            .rolling(window_size, min_periods=1)
            .mean()
        )
        self.windowed_df[var_cols] = (
            self.df[self.config["float_features"]]
            .rolling(window_size, min_periods=1)
            .var()
        )
        return

    def window_binary(self, window_size, mode="mode"):
        if mode == "median":
            median_cols = [f for f in self.config["binary_features"]]

            self.windowed_df[median_cols] = (
                self.df[self.config["binary_features"]]
                .rolling(window_size, min_periods=1)
                .median()
            )
        elif mode == "mode":
            mode_cols = [f for f in self.config["binary_features"]]

            self.windowed_df[mode_cols] = (
                self.df[self.config["binary_features"]]
                .rolling(window_size, min_periods=1)
                .mode()
            )
        elif mode == "max":
            max_cols = [f for f in self.config["binary_features"]]

            self.windowed_df[max_cols] = (
                self.df[self.config["binary_features"]]
                .rolling(window_size, min_periods=1)
                .max()
            )
        return


def get_args():
    parser = argparse.ArgumentParser(
        description="window csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_path", help="where to find input csv file",
    )
    parser.add_argument("output_path", help="where to place the csv")
    parser.add_argument(
        "config", help="where to find input config file",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    w = Windowing(args.input_path, config=args.config)
    df = w.window_dataframe()
    df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
