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
        return

    def get_file_paths(self):
        self.feature_files = {}

        for fs in self.config["feature_sets"]:
            print(fs)
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
        for i in self.feature_files.values():
            assert (
                len(i) == self.num_examples
            ), "   all feature sets must have the same number of files"
        return

    def load_all_dataframes(
        self,
        df_as_list=False,
        force_reload=False,
        feather_dir="./data/feathered_data",
        rename_cols=False,
        test_on_two_examples=False,
    ):
        feather_path = f"{feather_dir}/{self.data_hash}.feather"
        # the data has is unique to the config, so if it exists it is identical
        # to what would be loaded from the CSV
        if exists(feather_path) and not force_reload and not test_on_two_examples:
            print("Reading from feather file")
            df = pd.read_feather(feather_path)
            return df, self.data_hash

        # Reloading all dataframes from CSV
        all_data_frames = []
        for i in range(self.num_examples):
            i_data_frames = []

            for fs, file_lists in self.feature_files.items():
                prefix = "".join([l[0] for l in fs.split("-")])
                cols = self.config["features"][fs]
                df_i = pd.read_csv(file_lists[i])[cols]
                if prefix != "AS" and rename_cols:  # (not our speaker annotations)
                    df_i.columns = [prefix + c for c in df_i.columns]
                i_data_frames.append(df_i)

            print("Shape of loaded frames:")
            for p in i_data_frames:
                print(p.shape)

            if test_on_two_examples and i > 0:
                break

            all_data_frames.append(pd.concat(i_data_frames, axis=1))

        if df_as_list:
            print("returning list")
            return all_data_frames, self.data_hash

        # Reshape into a massive df with all examples stacked
        df = pd.concat(all_data_frames, axis=0)
        print("Final shape: ", df.shape)
        print(df.columns)
        df = df.fillna(0)
        df.reset_index(inplace=True)

        # TODO: if directory does not exist, create directory
        if not test_on_two_examples:
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
    def apply_rolling_window(self, df, win_size, keep_old_features, feature_config):
        print("Applying rolling window, size: ", win_size)

        if win_size == 1:
            return df
        print("load config")
        with open(feature_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        if self.config["mean_features"]:
            print("Rolling mean features")
            for f in self.config["mean_features"]:
                assert f in df.columns, f"{f} not in {df.columns}"
                mean_col = f + "_mean"
                df[mean_col] = df[f].rolling(win_size, min_periods=1).mean()
                # TODO: fix drop duplicating
                # if not keep_old_features:
                #     df.drop([f], inplace=True, axis=1)
        if self.config["variance_features"]:
            print("Rolling var features")
            for f in self.config["variance_features"]:
                assert f in df.columns, f"{f} not in {df.columns}"
                var_col = f + "_var"
                df[var_col] = df[f].rolling(win_size, min_periods=1).var()
                # if not keep_old_features:
                #     df.drop([f], inplace=True, axis=1)
        if self.config["median_features"]:
            print("Rolling median features")
            for f in self.config["median_features"]:
                assert f in df.columns, f"{f} not in {df.columns}"
                med_col = f + "_median"
                df[med_col] = df[f].rolling(win_size, min_periods=1).median()
                # if not keep_old_features:
                #     df.drop([f], inplace=True, axis=1)
        if self.config["max_features"]:
            print("Rolling max features")
            for f in self.config["max_features"]:
                assert f in df.columns, f"{f} not in {df.columns}"
                max_col = f + "_max"
                df[max_col] = df[f].rolling(win_size, min_periods=1).max()
                # if not keep_old_features:
                #     df.drop([f], inplace=True, axis=1)
        if not keep_old_features:
            todrop = []
            for _, v in self.config.items():
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


class TimeSeriesDataset(Dataset):
    @timeit
    def __init__(
        self,
        df,
        overlap=False,
        shuffle=True,
        subsample_perc=75,
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
        self.subsample = 100 - subsample_perc

        print(f"Datset loaded with shape {self.df.shape}")
        print(f"Labels loaded with shape {self.labels.shape}")
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

        if self.subsample:  # Undersample
            todrop = []
            for i in range(len(self.train_ind)):
                keep = False
                for l in self.labels:
                    if self.labels[l].iloc[self.train_ind[i]]:
                        keep = True
                if np.random.randint(100) > self.subsample and not keep:
                    todrop.append(i)

            todrop = set(todrop)
            print(f"Discarding {100-self.subsample}% of negative examples", end=": ")
            print(f"Dropping {len(todrop)} out of {len(self.train_ind)}")
            for index in sorted(todrop, reverse=True):
                del self.train_ind[index]

        self.weight_labels()

        print("Indices Created:")
        print(f"test: [{min(self.test_ind)}-{max(self.test_ind)}], ", end="")
        print(f"val: [{min(self.val_ind)}-{max(self.val_ind)}], ", end="")
        print(f"train: [{min(self.train_ind)}-{max(self.train_ind)}]")
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
            self.weights.append(0.5 / train_perc)
            print(f"Dataset balance for {c}: {100*perc:.01f}% of {len(self.labels)}")
            print(f"Train: {100*train_perc:.01f}% of {len(self.train_ind)}, ", end="")
            print(f"Val: {100*val_perc:.01f}% of {len(self.val_ind)}, ", end="")
            print(f"Test: {100*test_perc:.01f}% of {len(self.test_ind)}")
        print("Wights are: ", self.weights)

    @timeit
    def get_sk_dataset(self, feather_dir="./data/feathered_data"):
        assert self.overlap is False, "Overlap must be false for sklearn"

        f = int(math.floor(self.df.shape[0] / self.window)) * self.window
        self.sk_data = self.df.values[:f, :].reshape(-1, self.df.shape[1] * self.window)

        print(f"Sk data new shape is {self.sk_data.shape} \n")
        new_val_ind = [int(i / self.window) for i in self.val_ind]
        new_test_ind = [int(i / self.window) for i in self.test_ind]
        new_train_ind = [int(i / self.window) for i in self.train_ind]

        print("Updated Indices:")
        print(f"test: [{min(new_test_ind)}-{max(new_test_ind)}], ", end="")
        print(f"val: [{min(new_val_ind)}-{max(new_val_ind)}], ", end="")
        print(f"train: [{min(new_train_ind)}-{max(new_train_ind)}]")

        # Sk data new shape is (1245, 275411)
        # array with shape (39343, 275411)

        X_val = self.sk_data[new_val_ind]

        X_test = self.sk_data[new_test_ind]

        X_train = self.sk_data[new_train_ind]

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
