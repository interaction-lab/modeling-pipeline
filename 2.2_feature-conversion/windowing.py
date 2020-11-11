import pandas as pd
import numpy as np
from collections import Counter
import yaml
import argparse
import sys


class Windowing:
    def __init__(self, csv, config="./windowing_config.yml"):
        with open(config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.df = pd.read_csv(csv)
        print(self.df)

    def window_dataframe(self, window_size=30, step=10):
        # Contains Cols = 1 iff exists a 1 in the window
        # Mode Cols: take most common value in the window
        # All other cols: if NaN for majority of window = NaN
        # Otherwise, if not NaN for majority, use median.
        self.windowed_df = pd.DataFrame()

        self.window_float(window_size)
        if self.config["binary_features"]:
            self.window_binary(window_size, mode="median")

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

    def window_binary(self, window_size, mode="median"):
        if mode == "median":
            median_cols = [f + "_median" for f in self.config["binary_features"]]

            self.windowed_df[median_cols] = (
                self.df[self.config["binary_features"]]
                .rolling(window_size, min_periods=1)
                .median()
            )
        elif mode == "mode":
            mode_cols = [f + "_mode" for f in self.config["binary_features"]]

            self.windowed_df[mode_cols] = (
                self.df[self.config["binary_features"]]
                .rolling(window_size, min_periods=1)
                .mode()
            )
        elif mode == "max":
            max_cols = [f + "_max" for f in self.config["binary_features"]]

            self.windowed_df[max_cols] = (
                self.df[self.config["binary_features"]]
                .rolling(window_size, min_periods=1)
                .max()
            )
        return


def main(args):
    parser = argparse.ArgumentParser(
        description="take raw openface and produce clean files",
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

    w = Windowing(args.input_path, config=args.config)
    df = w.window_dataframe()
    df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
