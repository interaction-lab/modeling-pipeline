import yaml
import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

import itertools
from tqdm import tqdm

# import timeit
import math
import random
from pipeline.common.file_utils import ensure_destination_exists
from pipeline.common.function_utils import timeit
from .data_to_df import LoadDF


# from sklearn.preprocessing import StandardScaler

# Needed to reproduce
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


class TimeSeriesDataset(Dataset):
    @timeit
    def __init__(
        self,
        df,
        val_df=None,
        test_df=None,
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
            labels (list, optional): List of columns with data labels. Defaults to ["speaking"].
            data_hash (str, optional): [description]. Defaults to "".
        """
        # It is not recommended to shuffle if overlapping here
        self.status = "training"
        self.overlap = overlap
        self.shuffle = shuffle
        self.data_hash = data_hash
        self.subsample = 100 - subsample_perc

        if val_df is None and test_df is None:
            k=9
            p1 = math.floor(df.shape[0] / k)
            p2 = p1*2
            test_df = df.iloc[:p1,:] 
            val_df = df.iloc[p1:p2,:]
            train_df = df.iloc[p2:,:]
        else:
            assert val_df is not None and test_df is not None, "Must provide both val and test"
            train_df = df
        assert len(train_df.columns) == len(val_df.columns) == len(test_df.columns), "all sets must have the same name"
        
        self.train_labels, self.train_df = self.prepare_df(train_df, labels)
        self.val_labels, self.val_df = self.prepare_df(val_df, labels)
        self.test_labels, self.test_df = self.prepare_df(test_df, labels)
        return

    def prepare_df(self,df, labels):
        # Check for invalid features
        assert df.isin([np.nan, np.inf, -np.inf]).sum().sum() == 0
        if "index" in df.columns:
            df = df.drop(["index"], axis=1)
        print(df.columns)
        return df[labels], df.drop(labels, axis=1) 


    def get_df_indices(self, df):
        if self.overlap:
            indices = list(range(len(df) - self.window))
        else:
            indices = list(range(0, len(df) - (self.window - 1), self.window))
        if self.shuffle:
            random.shuffle(indices)
        return indices


    @timeit
    def setup_dataset(self, window=1):
        self.window = window
        self.train_ind = self.get_df_indices(self.train_df)
        self.test_ind = self.get_df_indices(self.test_df)
        self.val_ind = self.get_df_indices(self.val_df)

        if self.subsample:  # Undersample
            todrop = []
            for i in range(len(self.train_ind)):
                keep = False
                for l in self.train_labels:
                    if self.train_labels[l].iloc[self.train_ind[i]]:
                        keep = True
                if np.random.randint(100) > self.subsample and not keep:
                    todrop.append(i)

            todrop = set(todrop)
            print(f"Discarding {100-self.subsample}% of negative examples", end=": ")
            print(f"Dropping {len(todrop)} out of {len(self.train_ind)}")
            for index in sorted(todrop, reverse=True):
                del self.train_ind[index]

        self.weight_labels()

        return

    @timeit
    def split_dataset(self, start=2, k=7): #depricated
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

        return

    def weight_labels(self):
        # Find label balance
        self.weights = []
        for c in self.train_labels.columns:

            perc = self.train_labels[c].sum() / len(self.train_labels)

            self.train_perc = self.train_labels[c].iloc[self.train_ind].sum() / len(self.train_ind)
            self.val_perc = self.val_labels[c].iloc[self.val_ind].sum() / len(self.val_ind)
            self.test_perc = self.test_labels[c].iloc[self.test_ind].sum() / len(self.test_ind)

            # We only use training class balance for determining weights
            # We want the balance to be 50% so if we have 25% pos examples, the weight should 
            # be 2x on the positive class (.5/.25=2)
            self.weights.append(0.5 / self.train_perc)
            
            print(f"Dataset balance for {c}: {100*perc:.01f}% of {len(self.train_labels)}")
            print(f"Train: {100*self.train_perc:.01f}% of {len(self.train_ind)}, ", end="")
            print(f"Val: {100*self.val_perc:.01f}% of {len(self.val_ind)}, ", end="")
            print(f"Test: {100*self.test_perc:.01f}% of {len(self.test_ind)}")
        print("Wights are: ", self.weights)

    def trans_to_sk_dataset(self, feather_dir="./data/feathered_data"):
        assert self.overlap is False, "Overlap must be false for sklearn"
        # TODO allow for larger datasets so overlap is acceptable
        self.sk_data_path_train = f"{feather_dir}/tmp_train.npy"
        self.sk_data_path_test = f"{feather_dir}/tmp_test.npy"
        self.sk_data_path_val = f"{feather_dir}/tmp_val.npy"

        last = int(math.floor(self.train_df.shape[0] / self.window)) * self.window
        np.save(self.sk_data_path_train,
            self.train_df.values[:last, :].reshape(-1, self.train_df.shape[1] * self.window),
        )

        last = int(math.floor(self.test_df.shape[0] / self.window)) * self.window
        np.save(self.sk_data_path_test,
            self.test_df.values[:last, :].reshape(-1, self.test_df.shape[1] * self.window),
        )

        last = int(math.floor(self.val_df.shape[0] / self.window)) * self.window
        np.save(self.sk_data_path_val,
            self.val_df.values[:last, :].reshape(-1, self.val_df.shape[1] * self.window),
        )

        self.sk_val_ind = [int(i / self.window) for i in self.val_ind]
        self.sk_test_ind = [int(i / self.window) for i in self.test_ind]
        self.sk_train_ind = [int(i / self.window) for i in self.train_ind]

        return

    @timeit
    def get_sk_dataset(self):
        assert self.status in [
            "training",
            "validation",
            "testing",
        ], "status must be testing, validation, or training"

        self.trans_to_sk_dataset()
        # Sk data new shape is (1245, 275411)
        # array with shape (39343, 275411)
        if self.status == "training":
            self.sk_data_train = np.load(self.sk_data_path_train)
            X = self.sk_data_train[self.sk_train_ind]
            Y = np.array(
                [self.train_labels.iloc[i + self.window - 1] for i in self.train_ind]
            )

        if self.status == "validation":
            self.sk_data_val = np.load(self.sk_data_path_val)
            X = self.sk_data_val[self.sk_val_ind]
            Y = np.array([self.val_labels.iloc[i + self.window - 1] for i in self.val_ind])

        if self.status == "testing":
            self.sk_data_test = np.load(self.sk_data_path_test)
            X = self.sk_data_test[self.sk_test_ind]
            Y = np.array([self.test_labels.iloc[i + self.window - 1] for i in self.test_ind])
        return X, Y

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
            return (
                torch.FloatTensor(
                    self.train_df.iloc[ind[index] : ind[index] + self.window].values
                ),  # xb
                torch.FloatTensor(self.train_labels.iloc[ind[index] + self.window - 1]),  # yb
            )
        elif self.status == "validation":
            ind = self.val_ind
            return (
                torch.FloatTensor(
                    self.val_df.iloc[ind[index] : ind[index] + self.window].values
                ),  # xb
                torch.FloatTensor(self.val_labels.iloc[ind[index] + self.window - 1]),  # yb
            )
        elif self.status == "testing":
            ind = self.test_ind
            return (
                torch.FloatTensor(
                    self.test_df.iloc[ind[index] : ind[index] + self.window].values
                ),  # xb
                torch.FloatTensor(self.test_labels.iloc[ind[index] + self.window - 1]),  # yb
            )



class KaggleDataset(Dataset):
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
            labels (list, optional): List of columns with data labels. Defaults to ["speaking"].
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

    def trans_to_sk_dataset(self, feather_dir="./data/feathered_data"):
        assert self.overlap is False, "Overlap must be false for sklearn"
        # TODO allow for larger datasets so overlap is acceptable

        last = int(math.floor(self.df.shape[0] / self.window)) * self.window

        self.sk_data_path = f"{feather_dir}/tmp.npy"
        self.sk_data_path = os.path.abspath(self.sk_data_path)
        ensure_destination_exists(self.sk_data_path)
        # with open(self.sk_data_path, 'w') as f:
        np.save(
            self.sk_data_path,
            self.df.values[:last, :].reshape(-1, self.df.shape[1] * self.window),
        )

        self.new_val_ind = [int(i / self.window) for i in self.val_ind]
        self.new_test_ind = [int(i / self.window) for i in self.test_ind]
        self.new_train_ind = [int(i / self.window) for i in self.train_ind]
        return

    @timeit
    def get_sk_dataset(self):
        assert self.status in [
            "training",
            "validation",
            "testing",
        ], "status must be testing, validation, or training"

        self.trans_to_sk_dataset()
        self.sk_data = np.load(self.sk_data_path)

        # Sk data new shape is (1245, 275411)
        # array with shape (39343, 275411)
        if self.status == "training":
            X = self.sk_data[self.new_train_ind]
            Y = np.array(
                [self.labels.iloc[i + self.window - 1] for i in self.train_ind]
            )

        if self.status == "validation":
            X = self.sk_data[self.new_val_ind]
            Y = np.array([self.labels.iloc[i + self.window - 1] for i in self.val_ind])

        if self.status == "testing":
            X = self.sk_data[self.new_test_ind]
            Y = np.array([self.labels.iloc[i + self.window - 1] for i in self.test_ind])
        return X, Y

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
            ),  # xb
            torch.FloatTensor(self.labels.iloc[ind[index] + self.window - 1]),  # yb
        )

if __name__ == "__main__":
    # Recommended to run with python -m pipeline.modeling.data_to_df
    LDF = LoadDF(
        "examples/walkthrough/walkthrough_configs/data_loader_pearson_config.yml"
    )
    df, data_hash = LDF.load_all_dataframes(force_reload=True)

    md = TimeSeriesDataset(df, overlap=False, labels=["speaking"])
    md.setup_dataset(window=20)
    md.get_sk_dataset()
