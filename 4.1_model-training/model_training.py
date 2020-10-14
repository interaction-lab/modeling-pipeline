import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier

import xgboost as xgb
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    plot_confusion_matrix,
    average_precision_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import Dataset, DataLoader
from torch import optim
from data_utils import MyDataset, DataLoading
from model_defs import TCNModel, RNNModel, LSTMModel, GRUModel
import sklearn
import pprint
import torch
import pandas as pd
import numpy as np
import optuna
import neptune
from neptunecontrib.api import log_table
import matplotlib.pyplot as plt
import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

sns.set()

print(torch.cuda.is_available())
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class EarlyStopping(object):
    # Credit to https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
    def __init__(self, name, mode="max", min_delta=0, patience=10, percentage=False):
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

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            print("NAN score, exiting")
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            print(f" {self.name} lost patience")
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
            )
        elif params["model"] == "xgb":
            self.model = MultiOutputClassifier(
                xgb.XGBClassifier(
                    objective="multi:softprob",
                    max_depth=params["max_depth"],
                    num_class=params["num_classes"],
                    booster=params["booster"],
                    learning_rate=params["learning_rate"],
                )
            )
        elif params["model"] == "knn":
            self.model = KNeighborsClassifier(
                n_neighbors=params["n_neighbors"], leaf_size=params["leaf_size"]
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
            )

    def train_and_eval_model(self):

        if self.params["model"] in ["xgb", "tree", "forest", "knn", "mlp"]:
            auc_roc = self.fit_sklearn_classifier()
        else:
            weights = self.params["class_weights"]
            weights = torch.FloatTensor(weights)
            weights = weights.float().to(dev)
            self.loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=weights)

            self.model = self.model.float().to(dev)
            self.data_loader = DataLoader(
                self.dataset, batch_size=self.params["batch_size"], shuffle=True
            )

            self.opt = optim.SGD(self.model.parameters(), lr=self.params["lr"])
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt)

            auc_roc = self.fit_torch_nn()

        return auc_roc

    def fit_torch_nn(self):
        val_report = []
        train_report = []
        test_report = []
        es_los = EarlyStopping(
            "val_loss", patience=self.params["patience"], min_delta=0.005, mode="min"
        )
        es_val_auc = EarlyStopping(
            "val_auc", patience=self.params["patience"], min_delta=0.002, mode="max"
        )
        stopping_backup = 0

        for epoch in range(self.params["epochs"]):

            # TRAINING
            self.model.train()
            self.dataset.status = "training"
            (
                (columns, train_metric_values),
                train_wa_auROC,
                train_wa_AP,
                train_loss,
            ) = self.run_epoch()
            train_report.append(train_metric_values)

            # EVALUATION
            self.model.eval()
            self.dataset.status = "validation"
            with torch.no_grad():
                (
                    (columns, val_metric_values),
                    val_wa_auROC,
                    val_wa_AP,
                    val_loss,
                ) = self.run_epoch()
                val_report.append(val_metric_values)

            if self.trial:
                self.trial.report(val_wa_auROC, step=epoch)
                stop_on_loss = es_los.step(val_loss)
                stop_on_auc = es_val_auc.step(val_wa_auROC)
                if stop_on_auc or stop_on_loss:
                    stopping_backup += 1
                else:
                    stopping_backup = 0

                if (stop_on_loss and stop_on_auc) or (
                    stopping_backup > 2 * self.params["patience"]
                ):
                    # TESTING
                    self.model.eval()
                    self.dataset.status = "testing"
                    with torch.no_grad():
                        (
                            (columns, test_metric_values),
                            test_wa_auROC,
                            test_wa_AP,
                            total_loss,
                        ) = self.run_epoch()
                        test_report = [test_metric_values]
                        reports = {
                            "train": train_report[-1],
                            "val": val_report[-1],
                            "test": test_report[-1],
                        }

                        self.log_reports(reports, columns)
                    return test_wa_auROC

            self.scheduler.step(val_loss)
            if self.verbose:
                print(
                    f"Epoch {epoch+0:03} | Train: Loss-{train_loss:.2f} auROC-{train_wa_auROC:.4f} AP-{train_wa_AP:.4f} | Val: Loss-{val_loss:.4f} auROC-{val_wa_auROC:.4f} AP-{val_wa_AP:.4f}"
                )

        # TESTING
        self.model.eval()
        self.dataset.status = "testing"
        with torch.no_grad():
            (
                (columns, test_metric_values),
                test_wa_auROC,
                test_wa_AP,
                total_loss,
            ) = self.run_epoch()
            test_report = [test_metric_values]
            reports = {
                "train": train_report[-1],
                "val": val_report[-1],
                "test": test_report[-1],
            }

            self.log_reports(reports, columns)
        return test_wa_auROC

    def run_epoch(self):
        total_loss = 0
        labels = []
        predictions = []

        for xb, yb in self.data_loader:
            batch_loss, batch_predictions = self.run_batch(xb.to(dev), yb.to(dev))
            total_loss += batch_loss

            labels.append(yb.numpy())
            predictions.append(batch_predictions.cpu().detach().numpy())

        total_loss = total_loss / len(self.data_loader)

        predictions = np.vstack(predictions)
        labels = np.vstack(labels)

        result, wa_auROC, wa_AP = self.eval_metrics(labels, predictions)
        return (
            self.split_results_to_list(result, wa_auROC, wa_AP),
            wa_auROC,
            wa_AP,
            total_loss,
        )

    def run_batch(self, xb, yb):
        out = self.model(xb)
        loss = self.loss_func(out, yb)
        out = torch.sigmoid(out)

        if self.dataset.status == "training":
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

        preds = torch.round(out)

        return loss.item(), preds

    def fit_sklearn_classifier(self):
        X_train, X_val, X_test, Y_test, Y_train, Y_val = self.dataset.get_dataset()

        self.model.fit(X_train, Y_train)

        train_wa_auROC, train_wa_AP = self.test_fit(X_train, Y_train, "train")
        val_wa_auROC, val_wa_AP = self.test_fit(X_val, Y_val, "val")
        test_wa_auROC, test_wa_AP = self.test_fit(X_test, Y_test, "test")
        print(
            "auROC:", train_wa_auROC, val_wa_auROC, test_wa_auROC,
        )
        print("mAP:", train_wa_AP, val_wa_AP, test_wa_AP)

        return test_wa_auROC

    def test_fit(self, X_set, Y_set, set_name):
        Y_pred = self.model.predict(X_set)
        report, wa_auROC, wa_AP = self.eval_metrics(Y_set, Y_pred, output_dict=True)
        neptune.send_metric(f"{set_name}_wa_auROC", wa_auROC)
        neptune.send_metric(f"{set_name}_wa_AP", wa_AP)
        for k, v in report.items():
            for k2, v2 in v.items():
                neptune.send_metric(f"{set_name}_{k}-{k2}", v2)
        return wa_auROC, wa_AP

    def eval_metrics(self, labels, preds, output_dict=True):
        y_pred_list = [a.squeeze().tolist() for a in preds]
        y_test = [a.squeeze().tolist() for a in labels]

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        wa_auROC = roc_auc_score(y_test, y_pred_list, average="weighted")
        wa_AP = average_precision_score(y_test, y_pred_list, average="weighted")

        report = classification_report(
            y_test,
            y_pred_list,
            output_dict=output_dict,
            target_names=self.params["target_names"],
        )
        return report, wa_auROC, wa_AP

    def split_results_to_list(self, results, auROC, mAP):
        columns = ["wa_auROC", "wa_AP"]
        values = [auROC, mAP]
        for k, v in results.items():
            for k2, v2 in v.items():
                columns.append(f"{k}-{k2}")
                values.append(v2)
        return columns, values

    def log_reports(self, reports_dict, columns):
        for k, v in reports_dict.items():
            for i in range(len(columns)):
                neptune.send_metric(f"{k}_{columns[i]}", v[i])


if __name__ == "__main__":
    data_loader = DataLoading()
    print("starting")
    df = data_loader.get_one_person_talking_dataset(sessions=(1, 2), hard=True)

    window = 6
    data = MyDataset(
        df,
        window=window,
        pad_out=True,
        num_labels=4,
        continuous_labels=False,
        labels_at_end=True,
        augment=True,
    )
    for s in ["training", "validation", "testing"]:
        data.status = s
        print(len(data))

    totals = [0, 0, 0, 0]
    for example in data.y_train:
        for i in range(4):
            totals[i] += example[i]

    print(totals)
    model_params = {
        "model": "tcn",
        "hidden": 5,
        "batch_size": 21,
        "window": window,
        "dropout": 0.25,
        "depth": 7,
        "kern_size": 2,
        #
        "loss_func": torch.nn.BCEWithLogitsLoss(),
        "lr": 0.01,
        "epochs": 3,
        "num_features": 89,
        "num_classes": 4,
        "target_names": ["silent", "talking", "about_to_talk", "almost_finished"],
    }
    trainer = ModelTraining(model_params, data, verbose=True)
    auc_roc = trainer.train_and_eval_model()
