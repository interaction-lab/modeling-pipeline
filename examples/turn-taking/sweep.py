import neptune
import optuna
import joblib
import matplotlib.pyplot as plt
import pandas as pd

import neptunecontrib.monitoring.optuna as opt_utils
from neptunecontrib.api import log_table
from optuna.samplers import TPESampler

from pipeline.modeling.data_utils import TransformDF
from pipeline.modeling.datasets import TimeSeriesDataset
from pipeline.modeling.data_to_df import LoadDF
from pipeline.common.function_utils import timeit
from pipeline.modeling.model_training import ModelTraining
from pipeline.common.optimize_pandas import optimize


@timeit
def add_turn_labels(df, window, visualize=False):
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


def log_reports(metrics, columns, log_to_neptune, verbose=False):
    # Here is where we can get creative showing what we want
    for k in ["train", "val", "test"]:
        if k is "test" or model in ["forest", "tree", "mlp", "knn", "xgb"]:
            df = pd.DataFrame([metrics[k]], columns=columns)
        else:
            df = pd.DataFrame(metrics[k], columns=columns)
        if log_to_neptune:
            log_table(k, df)
        else:
            print(k, df)
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
                    if log_to_neptune:
                        neptune.send_metric(f"{k}_{c}", df[c].iloc[-1])
                    elif verbose:
                        print(f"{k}_{c}", df[c].iloc[-1])
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
        columns_to_plot = [
            c for c in df_train.columns if ("auROC" in c or "AP" in c or "f1" in c)
        ]
        # Create a figure showing metrics progress while training
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))
        df_train[columns_to_plot].plot(ax=axes[0])
        df_train["loss"].plot(ax=axes[0], secondary_y=True, color="black")
        axes[0].set_title("train")

        df_val[columns_to_plot].plot(ax=axes[1])
        df_val["loss"].plot(ax=axes[1], secondary_y=True, color="black")
        axes[1].set_title("val")
        if verbose:
            plt.show()
        if log_to_neptune:
            experiment.log_image("diagrams", fig)
    return


def set_model_params(trial, model):
    model_params = {}

    if model in ["tree", "forest"]:
        model_params = {
            # Tree/Forest Params
            "max_depth": trial.suggest_int("max_depth", 3, 18),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "min_samples_leaf": trial.suggest_loguniform("min_samples", 1e-4, 0.5),
            "window": trial.suggest_int("window", 1, MAX_HISTORY),
            "max_features": trial.suggest_categorical(
                "max_features", ["auto", "sqrt", "log2"]
            ),
        }
    if model in ["knn"]:
        model_params = {
            # KNN params
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 16),
            "leaf_size": trial.suggest_int("leaf_size", 15, 45),
            "window": trial.suggest_int("window", 1, MAX_HISTORY),
        }
    if model in ["xgb"]:
        model_params = {
            "max_depth": trial.suggest_int("max_depth", 3, 13),
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            "window": trial.suggest_int("window", 1, MAX_HISTORY),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.2),
        }
    if model in ["mlp"]:
        model_params = {
            # MLP Params
            "width": trial.suggest_int("width", 10, 100),
            "depth": trial.suggest_int("depth", 2, 7),
            "window": trial.suggest_int("window", 1, MAX_HISTORY),
            "activation": trial.suggest_categorical(
                "activation", ["logistic", "tanh", "relu"]
            ),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "sgd", "adam"]),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 5e-1),
        }
    if model in ["tcn"]:
        model_params = {
            # TCN Params
            "num_layers": trial.suggest_int("num_layers", 2, 8),
            "lr": trial.suggest_loguniform("learning_rate", 5e-5, 5e-3),
            "batch_size": trial.suggest_int("batch_size", 5, 25),
            "window": trial.suggest_int("window", 3, MAX_HISTORY),
            "kern_size": trial.suggest_int("kern_size", 1, 5),
            "dropout": 0.25,
            "epochs": 200,
        }
    if model in ["rnn", "gru", "lstm"]:
        model_params = {
            # TCN Params
            "num_layers": trial.suggest_int("num_layers", 2, 8),
            "lr": trial.suggest_loguniform("learning_rate", 5e-6, 5e-3),
            "batch_size": trial.suggest_int("batch_size", 5, 25),
            "window": trial.suggest_int("window", 3, MAX_HISTORY),
            "kern_size": trial.suggest_int("kern_size", 2, 7),
            "dropout": 0.25,
            "epochs": 200,
        }

    model_params["patience"] = PATIENCE
    model_params["weight_classes"] = WEIGHT_CLASSES
    model_params["model"] = model
    model_params["class_names"] = params["classes"]
    model_params["num_classes"] = len(model_params["class_names"])
    model_params["classes_org"] = LABELS_CLASSES

    return model_params


@timeit
def transform_dataset(trial, df, model_params):
    print("\n\n*****Transforming Dataset*******")
    tdf = TransformDF()
    rolling_window_size = trial.suggest_int("r_win_size", 1, MAX_FEATURE_ROLL)
    step_size = trial.suggest_int("step_size", 1, 6)

    df = tdf.apply_rolling_window(
        df,
        rolling_window_size,
        KEEP_UNWINDOWED_FEATURES,
        rw_config_path,
        # ALL_CLASSES,
    )
    df = tdf.sub_sample(df, step_size)
    if NORMALIZE:
        df = tdf.normalize_dataset(df, ALL_CLASSES)

    subsample_perc = trial.suggest_int("sub_sample_neg_perc", 50, 95)

    dataset = TimeSeriesDataset(
        df,
        labels=ALL_CLASSES,
        shuffle=SHUFFLE,
        subsample_perc=subsample_perc,
        # data_hash=FILE_HASH,
    )
    dataset.setup_dataset(window=model_params["window"])
    model_params["rolling_window_size"] = rolling_window_size
    model_params["step_size"] = step_size
    model_params["subsample_perc"] = subsample_perc
    model_params["num_features"] = dataset.df.shape[1]
    model_params["class_weights"] = dataset.weights
    if LOG_TO_NEPTUNE:
        neptune.send_metric("dataset_shape", df.shape[0])
    return dataset, model_params


@timeit
def restructure_and_save_data(exp_config_base):
    df_list, df_list_of_lists = [], []

    rw_config_path = f"{exp_config_base}/windowing.yml"

    # DF columns: left, right, center, bot
    if "left" in ALL_CLASSES:
        rw_config_path = f"{exp_config_base}/group/windowing.yml"
        dl_config_path = f"{exp_config_base}/group/data_loader.yml"

        data_loader = LoadDF(dl_config_path)
        df, _ = data_loader.load_all_dataframes(
            rename_cols=True, test_on_two_examples=SHORT_TEST
        )

    # DF columns: speaking
    else:
        for p in ["left", "right", "center"]:
            dl_config_path = f"{exp_config_base}/data_loader_{p}.yml"
            data_loader = LoadDF(dl_config_path)
            df_list_subset, _ = data_loader.load_all_dataframes(
                df_as_list=True, test_on_two_examples=SHORT_TEST
            )

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
        add_turn_labels(LOADED_DF, PREDICTION_WINDOW)
        LOADED_DF.drop(["index", "temp", "temp2"], inplace=True, axis=1)

    if exp_type == "predict":
        LOADED_DF[ALL_CLASSES] = LOADED_DF[ALL_CLASSES].shift(
            PREDICTION_WINDOW, fill_value=0
        )

    # Clean up speaking label if not used
    if "speaking" not in ALL_CLASSES and "speaking" in LOADED_DF.columns:
        LOADED_DF.drop(["speaking"], inplace=True, axis=1)

    LOADED_DF.to_feather(FDF_PATH)
    LOADED_DF = None  # Clear the space

    return data_loader, rw_config_path, dl_config_path


@timeit
def objective(trial):
    model_params = set_model_params(trial, model)

    df = pd.read_feather(FDF_PATH)

    dataset, model_params = transform_dataset(trial, df, model_params)

    print(f"\n\n******Training and evaluating {model}******")
    trainer = ModelTraining(model_params, dataset, trial, verbose=True)
    print(f"Traing and eval on data (size={dataset.df.shape}")
    summary_metric = trainer.train_and_eval_model()

    print("Logging reports")
    log_reports(trainer.metrics, trainer.columns, LOG_TO_NEPTUNE)

    return summary_metric


# ********************************************************************************
# *****************PARAMETERS TO CUSTOMIZE****************************************
# These parameters are used for controlling the sweep as
# well as describing the experiment for tracking in Neptune
# ********************************************************************************
EXP_NAME = "turn-taking"
COMPUTER = "personal-laptop"


# Current models ["tree", "forest", "xgb", "gru", "rnn", "lstm", "tcn", "mlp"]
models_to_try = [
    "xgb",
    "forest",
    "tcn",
    "tree",
    "rnn",
    "lstm",
    "gru",
]  # Not working: "mlp", "knn"

NUM_TRIALS = 2  # Number of trials to search for each model
PATIENCE = 2  # How many bad epochs to run before giving up
SHORT_TEST = True

# Each class should be a binary column in the df
# ACTIVE SPEAKER - is this person speaking at a given time step:
#    On a per face level:
#      "active_speaker": ["speaking"],
#    Classifying all four at once:
#      "active_speaker_l": ["left"],
#      "active_speaker_r": ["right"],
#      "active_speaker_c": ["center"],
#      "active_speaker_b": ["bot"],

# PREDICT SPEAKER - will this person speak at a future timestep:
#    On a per face level:
#      "predict_speaker": ["speaking"],
#    Classifying all four at once:
#      "predict_speaker_l": ["left"],
#      "predict_speaker_r": ["right"],
#      "predict_speaker_c": ["center"],
#      "predict_speaker_b": ["bot"],

# TURN TAKING - What is this person doing now?:
#    On a per face level:
#       "turns": ["taking", "yielding", "holding", "listening"],
#    Classifying all four at once:
#       TBD


# Other classification problems TBD
#   "uttertype": ["disclosure", "backchannel", "listening"]

# exp_type = "active"
# exp_type = "predict"
exp_type = "turn"

LABELS_CLASSES = {"turns": ["taking", "yielding", "holding", "listening"]}

# List of all classes
ALL_CLASSES = [i for _, v in LABELS_CLASSES.items() for i in v]

WEIGHT_CLASSES = True  # Weight loss against class imbalance
KEEP_UNWINDOWED_FEATURES = False

FEATURES = "pearson-a_hand-v"  # handcrafted, pearson, etc.

OVERLAP = False  # Should examples be allowed to overlap with each other
# when data includes multiple frames
SHUFFLE = False
NORMALIZE = True  # Normalize entire dataset (- mean & / std dev)
MAX_HISTORY = 30  # Max window the model can look through
PREDICTION_WINDOW = 30
MAX_FEATURE_ROLL = 30

LOG_TO_NEPTUNE = False


# ***********************************************************************************
# *****************Setup Experimental Details****************************************
# Load the data here so it is not reloaded in each call to
# optimize().
# Set up experimental parameters to be shared with neptune.
# Hyperparameters are set (and recorded) in optimize().
# ***********************************************************************************
FDF_PATH = "./data/feathered_data/tmp.feather"

LOAD_FROM_CSV = True

exp_config_base = f"./examples/{EXP_NAME}/configs/{FEATURES}"
if LOAD_FROM_CSV:
    print("reloading data!")
    data_loader, rw_config_path, dl_config_path = restructure_and_save_data(
        exp_config_base
    )
else:
    # We will reuse data loaded before, but we do need to know the config
    # so we grab a sample config
    rw_config_path = f"{exp_config_base}/windowing.yml"
    dl_config_path = f"{exp_config_base}/group/data_loader.yml"
    data_loader = LoadDF(dl_config_path)

# Record experimental details for Neptune
params = {
    "trials": f"{NUM_TRIALS}",
    "pruner": "no pruning",  # See optuna.create_study
    "classes": ALL_CLASSES,
    "patience": PATIENCE,
    "weight classes": WEIGHT_CLASSES,
    "overlap": OVERLAP,
    "shuffle": SHUFFLE,
    "normalize": NORMALIZE,
    "max_rolling": MAX_FEATURE_ROLL,
    "pred_window": PREDICTION_WINDOW,
}

tags = [
    COMPUTER,
    FEATURES,
    exp_type + ":" + "_".join(ALL_CLASSES),
    f"{NUM_TRIALS} Trials",
    f"{data_loader.num_examples} Sessions",
]


# ***************************************************************************
# *****************Run The Experiment****************************************
# Here were try and optimize the hyperparams of each
# model we are training
# ***************************************************************************
# Start up Neptune, init call takes the name of the sandbox
# Neptune requires that you have set your api key in the terminal
if LOG_TO_NEPTUNE:
    neptune.init(f"cmbirmingham/{EXP_NAME}")
    neptune_callback = opt_utils.NeptuneCallback(log_study=True, log_charts=True)

for model in models_to_try:
    tags.append(model)
    # folder_location = "./data/studies/study_{}_{}.pkl".format(model, EXP_NAME)
    sampler = TPESampler(seed=10)  # Needed for reproducing results

    print(f"***********Creating study for {model} ***********")
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.NopPruner(), sampler=sampler
    )
    if LOG_TO_NEPTUNE:
        experiment = neptune.create_experiment(
            name=f"{model}_{EXP_NAME}",
            params=params,
            upload_source_files=[
                "sweep.py",
                # "model_training.py",
                # "model_defs.py",
                # "data_utils.py",
                # config,
            ],
        )
        for t in tags:
            neptune.append_tag(t)
        study.optimize(objective, n_trials=NUM_TRIALS, callbacks=[neptune_callback])
        neptune.stop()
    else:
        study.optimize(objective, n_trials=NUM_TRIALS)
    tags.remove(model)
