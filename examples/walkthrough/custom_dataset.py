
import matplotlib.pyplot as plt
import pandas as pd
from pipeline.modeling.data_utils import TransformDF
from pipeline.modeling.datasets import TimeSeriesDataset
from pipeline.common.function_utils import timeit
from pipeline.common.optimize_pandas import optimize
from pipeline.modeling.data_to_df import LoadDF



class MakeTurnsDataset():
    def __init__(self, config_pth, classes, max_roll, keep_unwindowed, normalize, fdf_path) -> None:
        self.classes = classes
        self.config_pth = config_pth
        self.max_roll = max_roll
        self.keep_unwindowed = keep_unwindowed
        self.normalize = normalize
        self.fdf_path = fdf_path

        self.class_list = [i for _, v in classes.items() for i in v]

        data_loader = LoadDF(config_pth)
        LOADED_DF, FILE_HASH = data_loader.load_all_dataframes()

        if "turns" in classes.keys():
            self.add_turn_labels(LOADED_DF, 30)
            LOADED_DF.drop(["index", "temp", "temp2"], inplace=True, axis=1)

        if "speech" not in classes.keys():
            LOADED_DF.drop(["speaking"], inplace=True, axis=1)

        LOADED_DF.to_feather(fdf_path)

    @timeit
    def add_turn_labels(self, df, window, visualize=False):
        print("******Adding in turn taking labels*****")
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)

        df["yielding"] = 0.0
        df["temp"] = df[["speaking"]].rolling(window=indexer, min_periods=1).sum()
        df["temp"][df["temp"] < window] = 1.0
        df["yielding"][(df["speaking"] == 1.0) & (df["temp"] == 1.0)] = 1.0

        df["taking"] = 0.0
        df["temp2"] = df[["speaking"]].rolling(window=indexer, min_periods=1).sum()
        df["taking"][(df["speaking"] == 0) & (df["temp2"] > 0)] = 1.0

        df["holding"] = 0.0
        df["holding"][(df["speaking"] == 1) & (df["yielding"] == 0)] = 1.0

        df["listening"] = 0.0
        df["listening"][(df["speaking"] == 0) & (df["taking"] == 0)] = 1.0

        if visualize:
            df[["temp2", "speaking"]].plot.line()
            df[["speaking", "taking", "yielding"]].plot.line()
            plt.show()
        return

    @timeit
    def transform_dataset(self, trial, df, model_params, shuffle, window_config):
        print("\n\n*****Transforming Dataset*******")
        tdf = TransformDF()
        rolling_window_size = trial.suggest_int("r_win_size", 1, self.max_roll)
        step_size = trial.suggest_int("step_size", 1, 6)

        df = tdf.apply_rolling_window(
            df,
            rolling_window_size,
            self.keep_unwindowed,
            window_config,
        )
        df = tdf.sub_sample(df, step_size)
        if self.normalize:
            df = tdf.normalize_dataset(df, self.class_list)

        subsample_perc = trial.suggest_int("sub_sample_neg_perc", 50, 95)

        dataset = TimeSeriesDataset(
            df,
            labels=self.class_list,
            shuffle=shuffle,
            subsample_perc=subsample_perc,
            # data_hash=FILE_HASH,
        )
        dataset.setup_dataset(window=model_params["window"])
        model_params["rolling_window_size"] = rolling_window_size
        model_params["step_size"] = step_size
        model_params["subsample_perc"] = subsample_perc
        model_params["num_features"] = dataset.df.shape[1]
        model_params["class_weights"] = dataset.weights
        return dataset, model_params


    @timeit
    def restructure_and_save_data(self, exp_config_base, short_test, exp_type, pred_window):
        df_list, df_list_of_lists = [], []

        config_pth = f"{exp_config_base}/windowing.yml"

        # DF columns: left, right, center, bot
        if "left" in self.classes:
            config_pth = f"{exp_config_base}/group/windowing.yml"
            dl_config_path = f"{exp_config_base}/group/data_loader.yml"

            data_loader = LoadDF(dl_config_path)
            df, _ = data_loader.load_all_dataframes(prefix_col_names=True)

        # DF columns: speaking
        else:
            for p in ["left", "right", "center"]:
                dl_config_path = f"{exp_config_base}/data_loader_{p}.yml"
                data_loader = LoadDF(dl_config_path)
                df_list_subset, _ = data_loader.load_all_dataframes()

                # Convert lrc speaking labels to "speaking" column headers
                for df in df_list_subset:
                    c = list(df.columns)
                    c.remove(p)
                    c.append("speaking")
                    df.columns = c

                df_list_of_lists.append(df_list_subset)

            # Reorder dataframes
            for j in range(len(df_list_of_lists[0])):
                for i in range(3):
                    df_list.append(df_list_of_lists[i][j])

            # Join entire list of dataframes
            df = pd.concat(df_list, axis=0)

        # Compress and optimize dataframe
        LOADED_DF = optimize(df)

        LOADED_DF = LOADED_DF.fillna(0)
        LOADED_DF.reset_index(inplace=True)

        # Change labels for turn taking or prediction
        if exp_type == "turn":
            self.add_turn_labels(LOADED_DF, pred_window)
            LOADED_DF.drop(["index", "temp", "temp2"], inplace=True, axis=1)

        if exp_type == "predict":
            LOADED_DF[self.classes] = LOADED_DF[self.classes].shift(
                pred_window, fill_value=0
            )

        # Clean up speaking label if not used
        if "speaking" not in self.classes and "speaking" in LOADED_DF.columns:
            LOADED_DF.drop(["speaking"], inplace=True, axis=1)

        LOADED_DF.to_feather(self.fdf_path)
        LOADED_DF = None  # Clear the space

        return data_loader, config_pth, dl_config_path
