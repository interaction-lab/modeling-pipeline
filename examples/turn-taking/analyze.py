import neptune
import optuna
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import neptunecontrib.monitoring.optuna as opt_utils
from neptunecontrib.api import log_table
from optuna.samplers import TPESampler

from pipeline.modeling.data_utils import TimeSeriesDataset, LoadDF, TransformDF, timeit
from pipeline.modeling.model_training import ModelTraining
from pipeline.common.optimize_pandas import optimize


@timeit
def add_finishing_label(df, window):
    print("******Adding in Finishing Label*****")
    df["finishing"] = 0.0
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)
    df["temp"] = df[["speaking"]].rolling(window=indexer, min_periods=1).sum()
    df["temp"][df["temp"] < window] = 1.0
    df["finishing"][(df["speaking"] == 1.0) & (df["temp"] == 1.0)] = 1.0
    return


def log_reports(metrics, columns):
    # Here is where we can get creative showing what we want
    for k in ["train", "val", "test"]:
        if k is "test" or model in ["forest", "tree", "mlp", "knn", "xgb"]:
            df = pd.DataFrame([metrics[k]], columns=columns)
        else:
            df = pd.DataFrame(metrics[k], columns=columns)
        log_table(k, df)
        metrics_to_log = [
            "auROC",
            "AP",
            "support",
            "precision",
            "recall",
            "f1-score",
            "loss",
            "conf",
        ]
        for c in columns:
            for m in metrics_to_log:
                if m in c:
                    neptune.send_metric(f"{k}_{c}", df[c].iloc[-1])
    if model in ["forest", "tree", "mlp", "knn", "xgb"]:
        df_train = pd.DataFrame([metrics["train"]], columns=columns)
        df_val = pd.DataFrame([metrics["val"]], columns=columns)
    else:
        df_train = pd.DataFrame(metrics["train"], columns=columns)
        df_val = pd.DataFrame(metrics["val"], columns=columns)
    if df_val.shape[1] == 1:
        # Don't plot single point graphs
        return
    else:
        columns_to_plot = [c for c in df_train.columns if ("auROC" in c or "AP" in c)]
        # Create a figure showing metrics progress while training
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))
        df_train[columns_to_plot].plot(ax=axes[0])
        df_train["loss"].plot(ax=axes[0], secondary_y=True, color="black")
        axes[0].set_title("train")

        df_val[columns_to_plot].plot(ax=axes[1])
        df_val["loss"].plot(ax=axes[1], secondary_y=True, color="black")
        axes[1].set_title("val")

        experiment.log_image("diagrams", fig)
    return


CLASSES = [
    # "speaking",
    "finishing",
]  # List of class labels, e.g. ["speaking", "finishing"], ["speaking"], ["finishing"]


# Features provide a label for describing how correlated features
#   have been removed. The list of features to include is placed
#   in a config file which matches the pattern:
#   "./config/data_loader_{FEATURES}_config.yml"
FEATURES = "pearson-ext"  # handcrafted, pearson, pearson-ext etc.

SHUFFLE = False
OVERLAP = False  # Should examples be allowed to overlap with each other
NORMALIZE = True  # Normalize entire dataset (- mean & / std dev)
MAX_WINDOW = 20  # Max window the model can look through
CLOSING_WINDOW = 45
ROLL_FEATURES = True
KEEP_UNWINDOWED_FEATURES = False
# Rename to history? 'window' usage is confusing


# ************************************************************
# *****************Setup Experimental Details*****************
# Load the data here so it is not reloaded in each call to
# optimize().
# Set up experimental parameters to be shared with neptune.
# Hyperparameters are set (and recorded) in optimize().
# ************************************************************
FDF_PATH = "./data/feathered_data/tmp.feather"
rolling_window_config = (
    f"./examples/turn-taking/configs/windowing_{FEATURES}_config.yml"
)
RELOAD_DATA = True
if RELOAD_DATA:
    df_list = []
    df_list_of_lists = []
    for p in ["left", "right", "center"]:
        print("loading", p)
        config = f"examples/turn-taking/configs/data_loader_{FEATURES}_config_{p}.yml"
        data_loader = LoadDF(config)
        df_list_subset, FILE_HASH = data_loader.load_all_dataframes(
            df_as_list=True, force_reload=True
        )

        for df in df_list_subset:
            c = list(df.columns)
            c.remove(p)
            c.append("speaking")
            df.columns = c

        df_list_of_lists.append(df_list_subset)

    for j in range(len(df_list_of_lists[0])):
        for i in range(3):
            df_list.append(df_list_of_lists[i][j])

    # input("continue?")

    print("Concat")
    total_df = pd.concat(df_list, axis=0)
    print("Optimize")
    LOADED_DF = optimize(total_df)
    print("Final shape: ", LOADED_DF.shape)
    LOADED_DF = LOADED_DF.fillna(0)
    LOADED_DF.reset_index(inplace=True)

    # input("continue?")

    if "finishing" in CLASSES:
        add_finishing_label(LOADED_DF, CLOSING_WINDOW)
        LOADED_DF.drop(["index", "temp"], inplace=True, axis=1)

    if "speaking" not in CLASSES:
        # print("Select only finishing")
        # LOADED_DF = LOADED_DF[LOADED_DF["speaking"] > 0]
        LOADED_DF.drop(["speaking"], inplace=True, axis=1)
        print(f"New shape: {LOADED_DF.shape}")

    LOADED_DF.reset_index(inplace=True, drop=True)

    # Check the labels
    # LOADED_DF[["speaking", "finishing"]].iloc[1000:4000].plot.line()
    # plt.show()

    # Save the df that has been loaded with feather for quick reloading
    LOADED_DF.to_feather(FDF_PATH)
    LOADED_DF = None  # Clear the space
else:
    config = f"examples/turn-taking/configs/data_loader_{FEATURES}_config_center.yml"
    data_loader = LoadDF(config)

df = pd.read_feather(FDF_PATH)
if "index" in df.columns:
    df.drop(["index"], inplace=True, axis=1)
df.reset_index(inplace=True, drop=True)
tdf = TransformDF()
rolling_window_size = 10
step_size = 5

if ROLL_FEATURES:
    tdf.apply_rolling_window(
        df,
        rolling_window_size,
        KEEP_UNWINDOWED_FEATURES,
        rolling_window_config,
        CLASSES,
    )


print("\nStepping")
tdf.sub_sample(df, step_size)

if NORMALIZE:
    tdf.normalize_dataset(df, CLASSES)

print("\nCreate Dataset")
# subsample_perc = trial.suggest_int("sub_sample_neg_perc", 50, 95)

dataset = TimeSeriesDataset(
    df,
    labels=CLASSES,
    shuffle=SHUFFLE,
    # subsample_perc=subsample_perc,
    # data_hash=FILE_HASH,
)

print(df)

print(df.columns)


# from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np

# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

feat_cols = df.columns[1:]

# pca = PCA(n_components=3)
# pca_result = pca.fit_transform(df[feat_cols].values)
# df["pca-one"] = pca_result[:, 0]
# df["pca-two"] = pca_result[:, 1]
# df["pca-three"] = pca_result[:, 2]
# print(
#     "Explained variation per principal component: {}".format(
#         pca.explained_variance_ratio_
#     )
# )
# plt.figure(figsize=(16, 10))
# sns.scatterplot(
#     x="pca-one",
#     y="pca-two",
#     hue="y",
#     palette=sns.color_palette("hls", 10),
#     data=df.loc[rndperm, :],
#     legend="full",
#     alpha=0.3,
# )
# plt.show()

N = 8000
df_subset = df.loc[2000 : 2000 + N, :].copy()

data_subset = df_subset[feat_cols].values

plt.figure(figsize=(16, 10))

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=250)
tsne_results = tsne.fit_transform(data_subset)

df_subset["tsne-2d-one - 250"] = tsne_results[:, 0]
df_subset["tsne-2d-two"] = tsne_results[:, 1]
ax1 = plt.subplot(1, 3, 1)
sns.scatterplot(
    x="tsne-2d-one - 250",
    y="tsne-2d-two",
    hue=CLASSES[0],
    palette=sns.color_palette("hls", 2),
    data=df_subset,
    legend="full",
    alpha=0.3,
)

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=400)
tsne_results = tsne.fit_transform(data_subset)

df_subset["tsne-2d-one - 400"] = tsne_results[:, 0]
df_subset["tsne-2d-two"] = tsne_results[:, 1]
ax1 = plt.subplot(1, 3, 2)
sns.scatterplot(
    x="tsne-2d-one - 400",
    y="tsne-2d-two",
    hue=CLASSES[0],
    palette=sns.color_palette("hls", 2),
    data=df_subset,
    legend="full",
    alpha=0.3,
)

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
tsne_results = tsne.fit_transform(data_subset)

df_subset["tsne-2d-one - 1000"] = tsne_results[:, 0]
df_subset["tsne-2d-two"] = tsne_results[:, 1]
ax1 = plt.subplot(1, 3, 3)
sns.scatterplot(
    x="tsne-2d-one - 1000",
    y="tsne-2d-two",
    hue=CLASSES[0],
    palette=sns.color_palette("hls", 2),
    data=df_subset,
    legend="full",
    alpha=0.3,
)


plt.show()
