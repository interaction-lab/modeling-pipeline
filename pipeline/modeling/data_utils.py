import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from pipeline.common.function_utils import timeit

random.seed(0)
np.random.seed(0)


class TransformDF:
    def __init__(self) -> None:
        pass

    @timeit
    def normalize_dataset(self, df, columns_to_exclude=[]):
        # self.scaler = StandardScaler()
        print("Normalizing df columns")
        for c in tqdm(df.columns):
            if c not in columns_to_exclude:
                df[c] = (df[c] - df[c].mean()) / df[c].std()
        return df

    @timeit
    def apply_rolling_window(self, df, win_size, keep_old_features, feature_config):
        print("Applying rolling window, size: ", win_size)

        if win_size == 1:
            return df
        print("load config")
        with open(feature_config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        if config["mean_features"]:
            print("Rolling mean features")
            for f in config["mean_features"]:
                assert f in df.columns, f"{f} not in {df.columns}"
                mean_col = f + "_mean"
                df[mean_col] = df[f].rolling(win_size, min_periods=1).mean()
                # TODO: fix drop duplicating
                # if not keep_old_features:
                #     df.drop([f], inplace=True, axis=1)
        if config["variance_features"]:
            print("Rolling var features")
            for f in config["variance_features"]:
                assert f in df.columns, f"{f} not in {df.columns}"
                var_col = f + "_var"
                df[var_col] = df[f].rolling(win_size, min_periods=1).var()
                # if not keep_old_features:
                #     df.drop([f], inplace=True, axis=1)
        if config["median_features"]:
            print("Rolling median features")
            for f in config["median_features"]:
                assert f in df.columns, f"{f} not in {df.columns}"
                med_col = f + "_median"
                df[med_col] = df[f].rolling(win_size, min_periods=1).median()
                # if not keep_old_features:
                #     df.drop([f], inplace=True, axis=1)
        if config["max_features"]:
            print("Rolling max features")
            for f in config["max_features"]:
                assert f in df.columns, f"{f} not in {df.columns}"
                max_col = f + "_max"
                df[max_col] = df[f].rolling(win_size, min_periods=1).max()
                # if not keep_old_features:
                #     df.drop([f], inplace=True, axis=1)
        if not keep_old_features:
            todrop = []
            for _, v in config.items():
                if v:
                    for c in v:
                        todrop.append(c)
            todrop = list(set(todrop))
            df.drop(todrop, axis=1, inplace=True)

        df.fillna(0, inplace=True)
        return df

    @timeit
    def sub_sample(self, df, step):
        print(f"Old shape: {df.shape} (step {step})")
        if step != 1:
            if "index" in df.columns:
                df.drop(["index"], inplace=True, axis=1)
            df.reset_index(inplace=True, drop=True)
            todrop = [i for i in range(len(df)) if (i % step)]
            df.drop(todrop, inplace=True)
        print(f"New shape: {df.shape}")
        return df
