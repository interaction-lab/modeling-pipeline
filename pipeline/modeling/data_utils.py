from os.path import join, exists
import yaml

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

import time
import itertools
from tqdm import tqdm

# import timeit
import math
import random
import hashlib

# from sklearn.preprocessing import StandardScaler

# Needed to reproduce
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed


class LoadDF:
    def __init__(self, config="./config/data_loader_config.yml"):
        with open(config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # We name our compressed datset using a hash of the features so we
        #   can reload it without having to recreate it from the csvs
        self.data_hash = hashlib.sha224((str(self.config)).encode("UTF-8")).hexdigest()

        print("Configuration hash:", self.data_hash)
        self.get_file_paths()

    def get_file_paths(self):
        self.feature_files = {}

        for fs in self.config["feature_sets"]:
            config = self.config["feature_files"][fs]
            dir_list = [join(*config["dir_pattern"])]

            for sub_dir, subs in config["substitutes"].items():
                assert (
                    sub_dir in config["dir_pattern"]
                ), f"substitution ({sub_dir}) must have a target match in the path {config['dir_pattern']}"

                new_dir_list = []
                for p in dir_list:
                    for new_dir in subs:
                        new_dir_list.append(p.replace(sub_dir, str(new_dir), 1))

                dir_list = new_dir_list
                self.num_examples = len(dir_list)

            self.feature_files[fs] = dir_list
        print("Files to load: ")
        print(self.feature_files)
        for i in self.feature_files.values():
            assert (
                len(i) == self.num_examples
            ), "   all feature sets must have the same number of files"

    def load_all_dataframes(self, feather_dir="./data/feathered_data"):
        feather_path = f"{feather_dir}/{self.data_hash}.feather"
        if exists(feather_path):
            self.sk_df = pd.read_feather(feather_path)
        all_data_frames = []
        for i in range(self.num_examples):
            i_data_frames = []

            for fs, file_lists in self.feature_files.items():
                cols = self.config["features"][fs]
                i_data_frames.append(pd.read_csv(file_lists[i])[cols])

            print("Shape of loaded frames:")
            for p in i_data_frames:
                print(p.shape)

            all_data_frames.append(pd.concat(i_data_frames, axis=1))
        df = pd.concat(all_data_frames, axis=0)
        print("Final shape: ", df.shape)
        df = df.fillna(0)
        df.reset_index(inplace=True)
        df.to_feather(feather_path)
        return df, self.data_hash


class TransformDF:
    def __init__(self):
        return

    @timeit
    def normalize_dataset(self, df, columns_to_exclude=[]):
        # self.scaler = StandardScaler()
        print("Normalizing df columns")
        self.columns = df.columns
        for c in tqdm(df.columns):
            if c not in columns_to_exclude:
                df[c] = (df[c] - df[c].mean()) / df[c].std()
        return df

    @timeit
    def apply_rolling_window(
        self, df, window_size, keep_old_features, feature_config, labels
    ):
        print("Applying rolling window, size: ", window_size)
        if keep_old_features:
            windowed_df = df
        else:
            windowed_df = df[labels]

        if window_size == 1:
            return df

        with open(feature_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        if self.config["mean_features"]:
            mean_cols = [f + "_mean" for f in self.config["mean_features"]]

            windowed_df[mean_cols] = (
                df[self.config["mean_features"]]
                .rolling(window_size, min_periods=1)
                .mean()
            )
        if self.config["variance_features"]:
            var_cols = [f + "_var" for f in self.config["variance_features"]]
            windowed_df[var_cols] = (
                df[self.config["variance_features"]]
                .rolling(window_size, min_periods=1)
                .var()
            )
        if self.config["median_features"]:
            median_cols = [f + "_median" for f in self.config["median_features"]]

            windowed_df[median_cols] = (
                df[self.config["median_features"]]
                .rolling(window_size, min_periods=1)
                .median()
            )
        if self.config["mode_features"]:
            mode_cols = [f + "_mode" for f in self.config["mode_features"]]

            windowed_df[mode_cols] = (
                df[self.config["mode_features"]]
                .rolling(window_size, min_periods=1)
                .mode()
            )
        if self.config["max_features"]:
            max_cols = [f + "_max" for f in self.config["max_features"]]

            windowed_df[max_cols] = (
                df[self.config["max_features"]]
                .rolling(window_size, min_periods=1)
                .max()
            )
        windowed_df = windowed_df.fillna(0)
        return windowed_df

    @timeit
    def sub_sample(self, df, step):
        print(f"Old shape: {df.shape} (step {step})")
        if step != 1:
            df = df.loc[df.index[np.arange(len(df)) % step == 1]]
        print(f"New shape: {df.shape}")
        return df


class TimeSeriesDataset(Dataset):
    @timeit
    def __init__(
        self,
        df,
        overlap=False,
        shuffle=True,
        labels=["speaking"],
        data_hash="",
    ):
        """Produces a dataset that can be used with pytorch or sklearn

        Args:
            df ([type]): [description]
            normalize (bool, optional): [description]. Defaults to True.
            overlap (bool, optional): [description]. Defaults to False.
            shuffle (bool, optional): [description]. Defaults to True.
            labels (list, optional): [description]. Defaults to ["speaking"].
            data_hash (str, optional): [description]. Defaults to "".
        """

        # Check for invalid features
        assert df.isin([np.nan, np.inf, -np.inf]).sum().sum() == 0
        if "index" in df.columns:
            df = df.drop(["index"], axis=1)

        self.labels = df[labels]
        self.df = df.drop(labels, axis=1)

        # It is not recommended to shuffle if overlapping here
        self.status = "training"
        self.overlap = overlap
        self.shuffle = shuffle
        self.data_hash = data_hash
        return

    @timeit
    def setup_dataset(self, window=1):
        self.window = window

        if self.overlap:
            self.indices = list(range(len(self.df) - self.window))
        else:
            self.indices = list(range(0, len(self.df) - (self.window - 1), self.window))

        if self.shuffle:
            random.shuffle(self.indices)

        self.split_dataset()

        return

    @timeit
    def split_dataset(self, start=2, k=7):
        # Create indices for splitting the dataset
        # TODO Fix running training with actual cross validation
        fold_size = math.floor(len(self.indices) / k)

        split_points = list(range(0, len(self.indices), fold_size - 1))

        folds = [
            self.indices[split_points[i] : min(len(self.indices), split_points[i + 1])]
            for i in range(k)
        ]

        self.test_ind = folds.pop(start)
        self.val_ind = folds.pop(start % (k - 1))

        flatten = itertools.chain.from_iterable
        self.train_ind = list(flatten(folds))

        self.weight_labels()
        return

    def weight_labels(self):
        # Find label balance
        self.weights = []
        for c in self.labels.columns:

            perc = self.labels[c].sum() / len(self.labels)
            train_perc = self.labels[c].iloc[self.train_ind].sum() / len(self.train_ind)
            val_perc = self.labels[c].iloc[self.val_ind].sum() / len(self.val_ind)
            test_perc = self.labels[c].iloc[self.test_ind].sum() / len(self.test_ind)

            # We only use training class balance for determining weights
            self.weights.append(1 / train_perc)

            print(f"\n{c} {perc}% of the time overall (len={len(self.labels)})")
            print(f"{c} {train_perc}% of the time in train (len={len(self.train_ind)})")
            print(f"{c} {val_perc}% of the time in val (len={len(self.val_ind)})")
            print(f"{c} {test_perc}% of the time in test (len={len(self.test_ind)}\n)")

    @timeit
    def get_sk_dataset(self, feather_dir="./data/feathered_data"):
        assert self.overlap is False, "Overlap must be false for sklearn"

        # Reuse datasets for faster sklearn performance
        # if exists(f"{feather_dir}/{self.data_hash}-{self.window}-sk.feather"):
        if False:
            self.sk_df = pd.read_feather(
                f"{feather_dir}/{self.data_hash}-{self.window}-sk.feather"
            )
        else:
            # Flatten dataframe by window for sklearn
            self.sk_df = pd.concat(
                [
                    pd.DataFrame(
                        self.df.values[w :: self.window],
                    )
                    for w in range(self.window)
                ],
                axis=1,
            )
            self.sk_df.columns = [f"c-{c}" for c in range(len(self.sk_df.columns))]
            self.sk_df.to_feather(
                f"{feather_dir}/{self.data_hash}-{self.window}-sk.feather"
            )

        new_val_ind = [int(i / self.window) for i in self.val_ind]
        X_val = np.array(self.sk_df.values[new_val_ind])

        new_test_ind = [int(i / self.window) for i in self.test_ind]
        X_test = np.array(self.sk_df.values[new_test_ind])

        new_train_ind = [int(i / self.window) for i in self.train_ind]
        X_train = np.array(self.sk_df.values[new_train_ind])

        Y_train = np.array(
            [self.labels.iloc[i + self.window - 1] for i in self.train_ind]
        )
        Y_val = np.array([self.labels.iloc[i + self.window - 1] for i in self.val_ind])
        Y_test = np.array(
            [self.labels.iloc[i + self.window - 1] for i in self.test_ind]
        )
        return X_train, X_val, X_test, Y_test, Y_train, Y_val

    def __len__(self):
        if self.status == "training":
            return len(self.train_ind)
        if self.status == "validation":
            return len(self.val_ind)
        if self.status == "testing":
            return len(self.test_ind)

    def __getitem__(self, index):
        assert self.status in [
            "training",
            "validation",
            "testing",
        ], "status must be testing, validation, or training"

        if self.status == "training":
            ind = self.train_ind
        elif self.status == "validation":
            ind = self.val_ind
        elif self.status == "testing":
            ind = self.test_ind

        return (
            torch.FloatTensor(
                self.df.iloc[ind[index] : ind[index] + self.window].values
            ),
            torch.FloatTensor(self.labels.iloc[ind[index] + self.window - 1]),
        )


if __name__ == "__main__":
    data_loader = DataLoading()
    df = data_loader.get_all_sessions()
    md = MyDataset(df, overlap=False, window=20, labels=["speaking"])
    md.data_hash = data_loader.data_hash
    md.get_dataset()
    # To visualize:
    # prof = ProfileReport(md.labels, minimal=True)
    # prof.to_file(output_file="labels.html")
    # DL = DataLoader(md, batch_size=3)
    # for xb, yb in DL:
    #     print(yb)
    #     print(yb.shape)
    #     input()
