from os.path import join, exists
from os import makedirs
import yaml
import hashlib
import pandas as pd
from pipeline.common.file_utils import ensure_destination_exists
from pipeline.common.file_utils import get_dir_list_from_dict
from pipeline.common.function_utils import timeit


class LoadDF:
    """Takes a dataset of a bunch of csvs and converts them into a single DataFrame

    This dataframe can be returned or saved as a feather database for fast load times.

    We name our compressed datset using a hash of the features in the dataset. This enables the
    identical dataset to be loaded if the features haven't changed.

    A dataset configuration file has three parts: feature_sets, feature_files, and features
        feature_sets: the types of features to be included
        feature_files: the paths to the csv files for each feature set
        features: the feature names (and csv column headers) for each feature set
    """

    def __init__(self, config_path, feather_dir="./data/feathered_data"):
        """Loads the file paths and creates a hash for the dataset

        Args:
            config_path (str): [path to configuration of the dataset to use].
        """
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.data_hash = hashlib.sha224((str(self.config)).encode("UTF-8")).hexdigest()
        self.feather_path = f"{feather_dir}/{self.data_hash}.feather"

        print("Configuration hash:", self.data_hash)
        self.feature_files = {}

        for fs in self.config["feature_sets"]:
            print(fs)
            config = self.config["feature_files"][fs]

            self.feature_files[fs] = get_dir_list_from_dict(config)
            self.num_examples = len(self.feature_files[fs])
        for i in self.feature_files.values():
            assert (
                len(i) == self.num_examples
            ), "   all feature sets must have the same number of files"
        return

    @timeit
    def load_all_dataframes(self, force_reload=False, prefix_col_names=False):
        """Load the configured dataset of csvs into a single dataframe.

        Store the dataframe in a feather db with a data hash of the config for the name.

        Args:
            force_reload (bool, optional): Defaults to False.
            prefix_col_names (bool, optional): Defaults to False.

        Returns:
            df: joined dataset
        """

        # the data has is unique to the config, so if it exists it is identical
        # to what would be loaded from the CSV
        if exists(self.feather_path) and not force_reload:
            return pd.read_feather(self.feather_path), self.data_hash
        # Reloading all dataframes from CSVs
        all_dfs = []
        for ex_indx in range(self.num_examples):  # All sessions/examples
            session_dfs = []

            for fs, file_lists in self.feature_files.items():  # All feature types
                # Loading this examples dataframe
                df = pd.read_csv(file_lists[ex_indx])[self.config["features"][fs]]

                # Renaming columns
                prefix = "".join([l[0] for l in fs.split("-")])
                if prefix != "AS" and prefix_col_names:
                    # (never rename speaker annotations)
                    df.columns = [prefix + c for c in df.columns]

                session_dfs.append(df)
                print(fs, df.shape)

            # Sessions are concatenated together by frame
            all_dfs.append(pd.concat(session_dfs, axis=1))

        # Reshape into a massive df with all sessions stacked
        # and in one massive index
        df = pd.concat(all_dfs, axis=0)
        df = df.fillna(0)
        df.reset_index(inplace=True)
        # print("Final shape: ", df.shape)
        # print(df.columns)

        ensure_destination_exists(self.feather_path)
        df.to_feather(self.feather_path)
        return df, self.data_hash


if __name__ == "__main__":
    # Recommended to run with python -m pipeline.modeling.data_to_df
    LDF = LoadDF(
        "examples/walkthrough/walkthrough_configs/data_loader_pearson_config.yml"
    )
    LDF.load_all_dataframes(force_reload=True)
