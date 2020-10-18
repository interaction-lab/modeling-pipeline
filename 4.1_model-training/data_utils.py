import pandas as pd
import json
import yaml

from os import listdir
from os.path import isfile, join, exists
import sys
import time
import itertools
import timeit
import math
import random
import hashlib
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy

from sklearn.preprocessing import StandardScaler


# Needed to reproduce
random.seed(101)
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


class DataLoading:
    def __init__(self, config="./data_loader_config.yml"):
        with open(config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # We name our compressed datset using a hash of the features so we can reload
        # it without having to recreate it from the csvs
        self.data_hash = hashlib.sha224(
            (str(self.config["sessions"]) + str(self.config["features"])).encode(
                "UTF-8"
            )
        ).hexdigest()

        print(self.data_hash)

    def get_individual_session(self, session_num, position):
        assert session_num in self.config["sessions"]
        # assert person in self.config["positions"]
        data_frames = []

        for fs in self.config["feature_sets"]:
            for f in self.config["feature_files"][fs]:
                file_name = join(self.config["data_dir"], fs, str(session_num), f)

                if "Video-OpenFace" == fs and position not in f:
                    continue
                elif "Annotation-Turns" == fs:
                    print(f"reading {file_name}")
                    annotations = pd.DataFrame(
                        {"speaking": pd.read_csv(file_name)[position]}
                    )
                    data_frames.append(annotations)
                else:
                    print(f"reading {file_name}")
                    data_frames.append(
                        pd.read_csv(file_name)[self.config["features"][fs]]
                    )
        return pd.concat(data_frames, axis=1)

    @timeit
    def get_group_session(self, session_num):

        group = [
            self.get_individual_session(session_num, p)
            for p in self.config["positions"]
        ]
        return pd.concat(group, axis=0)

    @timeit
    def get_all_sessions(self):
        # All sessions are either loaded if they have already been created
        # or created from individual sessions. If the sessions are being recreated
        # they are saved to save time on future loading.
        print("Loading Sessions")
        if exists(f"data/{self.data_hash}.feather"):
            df = pd.read_feather(f"data/{self.data_hash}.feather")
            if "finishing" not in df.columns:
                df = self.get_all_sessions(
                    df, closing_window=self.config["features"]["closing_window"]
                )
        else:
            sessions = [self.get_group_session(i) for i in self.config["sessions"]]
            print("Concatenating Sessions")
            df = pd.concat(sessions, axis=0).reset_index()
            df = self.get_turn_ending(
                df, closing_window=self.config["features"]["closing_window"]
            )
            df = df.fillna(0)
            df.to_feather(f"data/{self.data_hash}.feather")
            print(df.columns)
            print(f"Saved to data/{self.data_hash}.feather")
        return df

    @timeit
    def get_turn_ending(self, df, closing_window=45):
        about_to_finish = 0
        df["finishing"] = float(0)
        for i in tqdm(range(len(df) - closing_window)):
            if df["speaking"].iloc[i]:
                for j in range(closing_window):
                    if not df["speaking"].iloc[i + j]:
                        df["finishing"].iloc[i] = float(1)
                        continue
        print(df["finishing"])
        print(sum(df["finishing"]))
        return df

    @timeit
    def write_sessions(self, df):
        print("Writing Sessions")
        # df.to_hdf("test_file.hdf", "test", mode="w", complib="blosc", format="table")
        df.to_feather(f"data/{self.data_hash}.feather")
        print("Done")


class MyDataset(Dataset):
    @timeit
    def __init__(
        self,
        df,
        window=2,
        status: str = "training",
        overlap: bool = False,
        labels=["speaking"],
    ):
        self.labels = df[labels]
        # print(self.labels)
        # print(self.labels.shape)
        self.df = df.drop(["index"], axis=1)
        self.df = df.drop(labels, axis=1)
        # print(self.df)
        # print(self.labels)
        # print(f"Total frames: {len(self.df)}")
        self.window = window
        self.status = status
        self.overlap = overlap

        self.setup_dataset()

    @timeit
    def split(self, start=0, k=5):
        ind_l = len(self.indices)
        fold_size = math.floor(ind_l / k)

        split_points = list(range(0, ind_l, fold_size - 1))
        folds = [
            self.indices[split_points[i] : min(ind_l, split_points[i + 1])]
            for i in range(k)
        ]

        self.test_list = folds.pop(start)
        self.val_list = folds.pop(start % (k - 1))
        flatten = itertools.chain.from_iterable

        self.train_list = list(flatten(folds))

        # print(
        #     f" Total examples {sum([len(self.train_list), len(self.val_list), len(self.test_list)])}"
        # )

        return

    @timeit
    def setup_dataset(self):
        print("setting up dataset")
        # Split into train/val/test
        if self.overlap:
            self.indices = list(range(len(self.df)))
        else:
            self.indices = list(range(0, len(self.df), self.window))

        self.split()

        # Fit scalar on train
        # self.scaler = StandardScaler()

        # Augment
        pass

    def get_dataset(self):
        pass

    def __len__(self):
        if self.status == "training":
            return len(self.train_list)
        if self.status == "validation":
            return len(self.val_list)
        if self.status == "testing":
            return len(self.test_list)

    def __getitem__(self, index):
        # print(self.df[self.train_list[index] : self.train_list[index] + self.window])
        if self.status == "training":
            return (
                torch.FloatTensor(
                    self.df[
                        self.train_list[index] : self.train_list[index] + self.window
                    ].values
                ),
                torch.from_numpy(
                    np.asarray(self.labels.iloc[self.train_list[index] + self.window])
                ),
            )
        if self.status == "validation":
            return (
                torch.FloatTensor(
                    self.df[
                        self.val_list[index] : self.val_list[index] + self.window
                    ].values
                ),
                torch.from_numpy(
                    np.asarray(self.labels.iloc[self.val_list[index] + self.window])
                ),
            )
        if self.status == "testing":
            return (
                torch.FloatTensor(
                    self.df[
                        self.test_list[index] : self.test_list[index] + self.window
                    ].values
                ),
                torch.from_numpy(
                    np.asarray(self.labels.iloc[self.test_list[index] + self.window])
                ),
            )


if __name__ == "__main__":
    data_loader = DataLoading()
    df = data_loader.get_all_sessions()
    md = MyDataset(df, window=3, labels=["speaking"])
    DL = DataLoader(md, batch_size=3)
    for xb, yb in DL:
        print(yb)
        print(yb.shape)
        input()
