# from sklearn.utils import class_weight
import torch
from torch.utils.data import DataLoader
from torch import optim
import pandas as pd
import numpy as np
from data_utils import MyDataset, DataLoading, timeit

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    plot_confusion_matrix,
    average_precision_score,
)

import xgboost as xgb
from model_defs import TCNModel, RNNModel, LSTMModel, GRUModel

import matplotlib.pyplot as plt

import sys
from tqdm import tqdm

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


# Needed to reproduce
torch.manual_seed(0)
np.random.seed(0)

print(f"Cuda? {torch.cuda.is_available()}")
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class EarlyStopping(object):
    # Credit to https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
    def __init__(
        self, name, mode="min", min_delta=0.001, patience=10, percentage=False
    ):
        self.name = name
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metric):
        if self.best is None:
            self.best = metric
            return False

        if np.isnan(metric):
            print("NAN score, exiting")
            return True

        if self.is_better(metric, self.best):
            print("updating best:", metric)
            self.num_bad_epochs = 0
            self.best = metric
        else:
            print("thinning patience")
            self.num_bad_epochs += 1
        print(self.num_bad_epochs, self.patience)
        if self.num_bad_epochs >= self.patience:
            print(f"{self.name} lost patience")
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


class ModelTraining:
    def __init__(self, params, dataset, trial=None, verbose=True):
        self.trial = trial
        self.dataset = dataset
        self.verbose = verbose
        self.params = params
        print(f"starting round with these params: \n {params}")

        if params["model"] == "tcn":
            self.model = TCNModel(
                num_channels=[params["num_features"]] * params["num_layers"],
                window=params["window"],
                kernel_size=params["kern_size"],
                num_classes=params["num_classes"],
                dropout=params["dropout"],
            )
        elif params["model"] == "rnn":
            self.model = RNNModel(
                hidden_size=params["kern_size"],
                input_size=params["num_features"],
                num_layers=params["num_layers"],
                num_classes=params["num_classes"],
                dropout=params["dropout"],
            )
        elif params["model"] == "lstm":
            self.model = LSTMModel(
                input_size=params["num_features"],
                hidden_size=params["kern_size"],
                num_layers=params["num_layers"],
                num_classes=params["num_classes"],
                dropout=params["dropout"],
            )
        elif params["model"] == "gru":
            self.model = GRUModel(
                input_size=params["num_features"],
                hidden_size=params["kern_size"],
                num_layers=params["num_layers"],
                num_classes=params["num_classes"],
                dropout=params["dropout"],
            )
        elif params["model"] == "tree":
            self.model = DecisionTreeClassifier(
                max_depth=params["max_depth"],
                criterion=params["criterion"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                class_weight="balanced",
            )
        elif params["model"] == "forest":
            self.model = RandomForestClassifier(
                max_depth=params["max_depth"],
                criterion=params["criterion"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                class_weight="balanced",
                verbose=1,
            )
        elif params["model"] == "xgb":
            # Add single and multiclass options
            if params["num_classes"] > 1:
                self.model = MultiOutputClassifier(
                    xgb.XGBClassifier(
                        objective="multi:softprob",
                        max_depth=params["max_depth"],
                        num_class=params["num_classes"],
                        booster=params["booster"],
                        learning_rate=params["learning_rate"],
                        verbosity=1,
                    )
                )
            else:
                self.model = xgb.XGBClassifier(
                    objective="binary:logistic",
                    max_depth=params["max_depth"],
                    num_class=params["num_classes"],
                    booster=params["booster"],
                    learning_rate=params["learning_rate"],
                    verbosity=1,
                )
        elif params["model"] == "knn":
            self.model = KNeighborsClassifier(
                n_neighbors=params["n_neighbors"],
                leaf_size=params["leaf_size"],
                verbose=True,
            )
        elif params["model"] == "mlp":
            hidden_layer_sizes = tuple(
                [params["width"] // i for i in range(1, params["depth"] + 1)]
            )
            self.model = MLPClassifier(
                solver=params["solver"],
                activation=params["activation"],
                learning_rate="adaptive",
                learning_rate_init=params["learning_rate"],
                hidden_layer_sizes=hidden_layer_sizes,
                verbose=True,
            )

        self.metrics = {"train": [], "val": [], "test": []}
        return

    @timeit
    def train_and_eval_model(self):

        if self.params["model"] in ["xgb", "tree", "forest", "knn", "mlp"]:
            auc_roc = self.fit_sklearn_classifier()
        else:
            if self.params["weight_classes"]:
                weights = torch.FloatTensor(self.dataset.weights)
                weights = weights.float().to(dev)
                self.loss_func2 = torch.nn.BCELoss()
                self.loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
            else:
                self.loss_func = torch.nn.BCEWithLogitsLoss()

            self.model = self.model.float().to(dev)
            self.data_loader = DataLoader(
                self.dataset, batch_size=self.params["batch_size"], shuffle=True
            )

            self.opt = optim.SGD(self.model.parameters(), lr=self.params["lr"])
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt)

            auc_roc = self.fit_torch_nn()
        return auc_roc

    def fit_torch_nn(self):
        print(f"Fitting PyTorch Classifier - {self.params['model']}")
        es_los = EarlyStopping("val_loss", patience=self.params["patience"])

        for self.epoch in tqdm(range(self.params["epochs"])):

            # ******** TRAINING ********
            self.model.train()
            self.dataset.status = "training"

            train_metric_values, _ = self.run_epoch()

            self.metrics["train"].append(train_metric_values)
            loss_index = self.columns.index("loss")

            # ******** EVALUATION ********
            self.model.eval()
            self.dataset.status = "validation"

            with torch.no_grad():
                val_metric_values, val_summary = self.run_epoch()

                self.metrics["val"].append(val_metric_values)
                val_loss = val_metric_values[loss_index]

            if self.trial:
                self.trial.report(val_summary, step=self.epoch)

            # EARLY STOPPING CHECK
            stop_early = es_los.step(val_loss)

            # ******** TESTING ********
            if stop_early or self.epoch == self.params["epochs"] - 1:
                self.model.eval()
                self.dataset.status = "testing"
                with torch.no_grad():
                    self.metrics["test"], _ = self.run_epoch()
                return val_summary

            self.scheduler.step(val_summary)

    def run_epoch(self):
        total_loss = 0
        labels, predictions = [], []

        for xb, yb in tqdm(self.data_loader):
            batch_loss, batch_predictions = self.run_batch(xb.to(dev), yb.to(dev))
            total_loss += batch_loss

            labels.append(yb.numpy())
            predictions.append(batch_predictions.cpu().detach().numpy())

        avg_loss = total_loss / len(self.data_loader)

        report, summary_stat = self.calculate_metrics(
            np.vstack(labels), np.vstack(predictions)
        )

        m_list, _ = self.listify_metrics(report, avg_loss)
        s = self.dataset.status
        print(f"{self.epoch}-{s}: L: {avg_loss:.3f} | auROC {summary_stat:.3f}")

        return m_list, summary_stat

    def run_batch(self, xb, yb):
        raw_out = self.model(xb)
        # BCEwithlogits does the sigmoid
        loss = self.loss_func(raw_out, yb)
        batch_loss = loss.item()

        # BCEwithlogits occassionally returns a huge loss
        # we replace these losses
        if batch_loss > 100:
            loss_alt = self.loss_func2(torch.sigmoid(raw_out), yb)
            batch_loss_alt = loss_alt.item()
            # print(f"\nSwitching: {batch_loss} <- {batch_loss_alt}")
            loss = loss_alt
            batch_loss = batch_loss_alt

        if self.dataset.status == "training":
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            # if args.clip_gradient is not None:
            #     total_norm = clip_grad_norm(self.model.parameters(), args.clip_gradient)
            #     if total_norm > args.clip_gradient:
            #         print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))
            self.opt.step()
            self.opt.zero_grad()

        preds = torch.round(torch.sigmoid(raw_out))
        # if batch_loss > 200:
        #     print(f"\nLoss was {batch_loss}")
        #     print(f"Or: {self.loss_func2(torch.sigmoid(raw_out), yb).item()}")

        return batch_loss, preds

    def fit_sklearn_classifier(self):
        print(f"Fitting Sklearn Classifier - {self.params['model']}")
        X_train, X_val, X_test, Y_test, Y_train, Y_val = self.dataset.get_dataset()
        sample_weights = compute_sample_weight(class_weight="balanced", y=Y_train)

        self.model.fit(X_train, Y_train, sample_weight=sample_weights)
        self.metrics = {}

        self.metrics["train"], _, _ = self.test_fit(X_train, Y_train, "train")
        self.metrics["val"], _, val_result_summary = self.test_fit(X_val, Y_val, "val")
        self.metrics["test"], test_results_df, test_result_summary = self.test_fit(
            X_test, Y_test, "test"
        )
        print(f"Test results:")
        print(test_results_df)

        return val_result_summary

    def test_fit(self, X_set, Y_set, set_name):
        Y_pred = self.model.predict(X_set)
        metrics_dict, m_summary = self.calculate_metrics(
            Y_set, Y_pred, output_dict=True
        )
        m_list, m_df = self.listify_metrics(metrics_dict)
        return m_list, m_df, m_summary

    def calculate_metrics(self, labels, preds, output_dict=True):
        y_pred_list = [a.squeeze().tolist() for a in preds]
        y_test = [a.squeeze().tolist() for a in labels]
        if type(y_test[0]) is list:
            y_test = [[int(a) for a in b] for b in y_test]
        else:
            y_test = [int(a) for a in y_test]

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        try:
            auROC = roc_auc_score(
                y_test, y_pred_list, average=None
            )  # average="weighted")
        except Exception as e:
            print(e)
            print(f"labels: {y_test}")
            print(f"predictions: {y_pred_list}")
            raise e

        AvgPrec = average_precision_score(
            y_test, y_pred_list, average=None
        )  # average="weighted")

        report = classification_report(
            y_test,
            y_pred_list,
            output_dict=output_dict,
            labels=[i for i in range(len(self.params["target_names"]))],
            target_names=self.params["target_names"],
        )

        if type(auROC) is np.ndarray:
            for i in range(len(self.params["target_names"])):
                report[self.params["target_names"][i]]["auROC"] = auROC[i]
                report[self.params["target_names"][i]]["AP"] = AvgPrec[i]
            summary_stat = sum(auROC) / len(auROC)
        else:
            report[self.params["target_names"][0]]["auROC"] = auROC
            report[self.params["target_names"][0]]["AP"] = AvgPrec
            summary_stat = auROC
        return report, summary_stat

    def listify_metrics(self, metrics_dict, loss=0):
        columns = ["loss"]
        values = [loss]
        for k, v in metrics_dict.items():
            for k2, v2 in v.items():
                columns.append(f"{k}-{k2}")
                values.append(v2)
        self.columns = columns
        df = pd.DataFrame([values], columns=columns)
        return values, df

    def plot_metrics(self):
        df_train = pd.DataFrame(self.metrics["train"], columns=self.columns)
        df_val = pd.DataFrame(self.metrics["val"], columns=self.columns)

        if df_val.shape[1] == 1:
            return
        else:
            columns_to_plot = [
                c for c in df_train.columns if ("auROC" in c or "AP" in c)
            ]
            # val_columns_to_plot = [c for c in df_val.columns]
            # Support and loss

            # Create a figure showing metrics progress while training
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))
            df_train[columns_to_plot].plot(ax=axes[0])
            df_train["loss"].plot(ax=axes[0], secondary_y=True, color="black")
            axes[0].set_title("train")

            df_val[columns_to_plot].plot(ax=axes[1])
            df_val["loss"].plot(ax=axes[1], secondary_y=True, color="black")
            axes[1].set_title("val")
            plt.show()
            # experiment.log_image("diagrams", fig)


if __name__ == "__main__":
    print("Starting")
    model_params = {}
    data_loader = DataLoading()
    df = data_loader.get_all_sessions()

    window = 5
    # classes = ["speaking", "finishing"]
    classes = ["finishing"]
    data = MyDataset(df, window=window, overlap=False, labels=classes)

    model = "tcn"
    if model in ["tcn", "rnn", "gru", "lstm"]:
        model_params = {
            # TCN Params
            "num_layers": 4,
            "lr": 0.00001576805111634853,
            "batch_size": 30,
            "window": window,
            "dropout": 0.25,
            "epochs": 200,
            "kern_size": 3,
        }
    if model in ["tree", "forest"]:
        model_params = {
            # Tree/Forest Params
            "max_depth": 4,
            "criterion": "entropy",
            "min_samples_leaf": 1e-3,
            "max_features": "auto",
        }
    if model in ["xgb"]:
        model_params = {
            "max_depth": 5,
            "booster": "dart",
            "learning_rate": 0.01,
        }
    model_params["weight_classes"] = True

    model_params["model"] = model
    model_params["patience"] = 2
    model_params["target_names"] = classes
    model_params["num_classes"] = len(classes)
    model_params["num_features"] = data.df.shape[1]
    model_params["class_weights"] = data.weights

    trainer = ModelTraining(model_params, data, verbose=True)

    auc_roc = trainer.train_and_eval_model()
    trainer.plot_metrics()
