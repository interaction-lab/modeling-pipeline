import pandas as pd
import numpy as np
import json
import random

from os import listdir
from os.path import isfile, join, exists
import sys
import yaml
import time

import torch
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy
from sklearn.preprocessing import StandardScaler
import timeit
import hashlib


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
            # print(config)
        # self.data_hash = hashlib.sha224(b"Nobody repetition").hexdigest()
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
                if "Video-OpenFace" == fs and position not in f:
                    continue
                else:
                    file_name = join(self.config["data_dir"], fs, str(session_num), f)
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
        print("Loading Sessions")
        if exists(f"data/{self.data_hash}.feather"):
            df = pd.read_feather(f"data/{self.data_hash}.feather")
        else:
            sessions = [self.get_group_session(i) for i in self.config["sessions"]]
            print("Concatenating Sessions")
            df = pd.concat(sessions, axis=0).reset_index()
            df.to_feather(f"data/{self.data_hash}.feather")
            print(df.columns)
            print(f"Saved to data/{self.data_hash}.feather")
        return df

    @timeit
    def write_sessions(self, df):
        print("Writing Sessions")
        # df.to_hdf("test_file.hdf", "test", mode="w", complib="blosc", format="table")
        df.to_feather(f"data/{self.data_hash}.feather")
        print("Done")


if __name__ == "__main__":
    data_loader = DataLoading()
    df = data_loader.get_all_sessions()
    print(df.shape)
    # data_loader.write_sessions(df)
