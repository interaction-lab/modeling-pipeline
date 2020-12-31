import neptune
import optuna
import joblib
import matplotlib.pyplot as plt
import pandas as pd

import neptunecontrib.monitoring.optuna as opt_utils
from neptunecontrib.api import log_table
from optuna.samplers import TPESampler

from pipeline.modeling.data_utils import TimeSeriesDataset, LoadDF, TransformDF, timeit
from pipeline.modeling.model_training import ModelTraining


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


@timeit
def objective(trial):
    joblib.dump(study, folder_location)
    model_params = {}

    if model in ["tree", "forest"]:
        model_params = {
            # Tree/Forest Params
            "max_depth": trial.suggest_int("max_depth", 3, 18),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "min_samples_leaf": trial.suggest_loguniform("min_samples", 1e-3, 0.5),
            "window": trial.suggest_int("window", 1, MAX_WINDOW),
            "max_features": trial.suggest_categorical(
                "max_features", ["auto", "sqrt", "log2"]
            ),
        }
    if model in ["knn"]:
        model_params = {
            # KNN params
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 16),
            "leaf_size": trial.suggest_int("leaf_size", 15, 45),
            "window": trial.suggest_int("window", 1, MAX_WINDOW),
        }
    if model in ["xgb"]:
        model_params = {
            "max_depth": trial.suggest_int("max_depth", 5, 10),
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            "window": trial.suggest_int("window", 1, MAX_WINDOW),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.2),
        }
    if model in ["mlp"]:
        model_params = {
            # MLP Params
            "width": trial.suggest_int("width", 10, 100),
            "depth": trial.suggest_int("depth", 2, 7),
            "window": trial.suggest_int("window", 1, MAX_WINDOW),
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
            "lr": trial.suggest_loguniform("learning_rate", 5e-6, 5e-4),
            "batch_size": trial.suggest_int("batch_size", 5, 25),
            "window": trial.suggest_int("window", 3, MAX_WINDOW),
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
            "window": trial.suggest_int("window", 3, MAX_WINDOW),
            "kern_size": trial.suggest_int("kern_size", 2, 7),
            "dropout": 0.25,
            "epochs": 200,
        }

    try:
        # Transform dataset
        print("\n\n\n*****Transforming Dataset*******")
        tdf = TransformDF()
        rolling_window_size = trial.suggest_int("r_win_size", 1, 10)
        step_size = trial.suggest_int("step_size", 1, 3)

        df = tdf.apply_rolling_window(
            LOADED_DF,
            rolling_window_size,
            KEEP_UNWINDOWED_FEATURES,
            rolling_window_config,
            CLASSES,
        )
        df = tdf.sub_sample(df, step_size)
        if NORMALIZE:
            df = tdf.normalize_dataset(df, CLASSES)

        dataset = TimeSeriesDataset(df, labels=CLASSES, data_hash=FILE_HASH)

        dataset.setup_dataset(window=model_params["window"])

        model_params["patience"] = PATIENCE
        model_params["weight_classes"] = WEIGHT_CLASSES
        model_params["model"] = model
        model_params["target_names"] = params["classes"]
        model_params["num_features"] = dataset.df.shape[1]
        model_params["class_weights"] = dataset.weights
        model_params["num_classes"] = len(model_params["target_names"])

        print("\n\n\n******Training and evaluating model******")
        print("Creating Trainer:")
        trainer = ModelTraining(model_params, dataset, trial, verbose=True)
        print(f"Traing and eval on data (size={dataset.df.shape}")
        summary_metric = trainer.train_and_eval_model()

        print("Logging reports")
        log_reports(trainer.metrics, trainer.columns)

        return summary_metric
    except ValueError as E:
        print("Trial Failed!")
        print(E)
        return 0


# *********************************************************
# *****************PARAMETERS TO CUSTOMIZE*****************
# These parameters are used for controlling the sweep as
# well as describing the experiment for tracking in Neptune
# *********************************************************
EXP_NAME = "turn-taking"
COMPUTER = "cmb-laptop"

# Current models ["tree", "forest", "xgb", "gru", "rnn", "lstm", "tcn", "mlp"]
models_to_try = [
    "tree",
    "forest",
    "xgb",
    "rnn",
    "lstm",
    "gru",
    "tcn",
]  # Not working: "mlp", "knn"

NUM_TRIALS = 3  # Number of trials to search for each model
PATIENCE = 2  # How many bad epochs to run before giving up

CLASSES = [
    "speaking"
]  # List of class labels, e.g. ["speaking", "finishing"], ["speaking"], ["finishing"]
WEIGHT_CLASSES = True  # Weight loss against class imbalance
KEEP_UNWINDOWED_FEATURES = False

# Features provide a label for describing how correlated features
#   have been removed. The list of features to include is placed
#   in a config file which matches the pattern:
#   "./config/data_loader_{FEATURES}_config.yml"
FEATURES = "handcrafted"  # handcrafted, pearson, etc.

OVERLAP = False  # Should examples be allowed to overlap with each other
NORMALIZE = True  # Normalize entire dataset (- mean & / std dev)
MAX_WINDOW = 20  # Max window the model can look through
# Rename to history? 'window' usage is confusing


# ************************************************************
# *****************Setup Experimental Details*****************
# Load the data here so it is not reloaded in each call to
# optimize().
# Set up experimental parameters to be shared with neptune.
# Hyperparameters are set (and recorded) in optimize().
# ************************************************************
rolling_window_config = (
    f"./examples/turn-taking/configs/windowing_{FEATURES}_config.yml"
)
df_list = []
for p in ["left", "right", "center"]:
    config = f"examples/{EXP_NAME}/configs/data_loader_{FEATURES}_config_{p}.yml"
    data_loader = LoadDF(config)
    LOADED_DF, FILE_HASH = data_loader.load_all_dataframes()
    # print(LOADED_DF.shape)
    c = list(LOADED_DF.columns)
    c.remove(p)
    c.append("speaking")
    LOADED_DF.columns = c
    df_list.append(LOADED_DF)

LOADED_DF = pd.concat(df_list, axis=0)
# print(LOADED_DF)

# Record experimental details for Neptune
params = {
    "trials": f"{NUM_TRIALS}",
    "pruner": "no pruning",  # See optuna.create_study
    "classes": CLASSES,
    "patience": PATIENCE,
    "weight classes": WEIGHT_CLASSES,
    "overlap": OVERLAP,
    "normalize": NORMALIZE,
}
tags = [
    EXP_NAME,
    # f"{len(data_loader.config['sessions'])} sess",
    "not-stat-windowed",
    COMPUTER,
    FEATURES,
]

# Start up Neptune, init call takes the name of the sandbox
# Neptune requires that you have set your api key in the terminal
neptune.init(f"cmbirmingham/{EXP_NAME}")
neptune_callback = opt_utils.NeptuneCallback(log_study=True, log_charts=True)


# ****************************************************
# *****************Run The Experiment*****************
# Here were try and optimize the hyperparams of each
# model we are training
# ****************************************************
for model in models_to_try:
    tags.append(model)
    print(f"***********Creating study for {model} ***********")
    experiment = neptune.create_experiment(
        name=f"{model}_{EXP_NAME}",
        params=params,
        upload_source_files=[
            "sweep.py",
            # "model_training.py",
            # "model_defs.py",
            # "data_utils.py",
            config,
            rolling_window_config,
        ],
    )
    folder_location = "./data/studies/study_{}_{}.pkl".format(model, EXP_NAME)
    for t in tags:
        neptune.append_tag(t)
    sampler = TPESampler(seed=10)  # Needed for reproducing results
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.NopPruner(), sampler=sampler
    )
    study.optimize(objective, n_trials=NUM_TRIALS, callbacks=[neptune_callback])
    neptune.stop()
    tags.remove(model)
