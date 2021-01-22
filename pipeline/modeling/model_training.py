# from sklearn.utils import class_weight
import torch
from torch.utils.data import DataLoader
from torch import optim
import pandas as pd
import numpy as np
from .data_utils import TimeSeriesDataset, LoadDF, timeit

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
from .model_defs import TCNModel, RNNModel, LSTMModel, GRUModel

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
            hyperopt_metric = self.fit_sklearn_classifier()
        else:
            if self.params["weight_classes"]:
                weights = torch.FloatTensor(self.dataset.weights)
                weights = weights.float().to(dev)
                self.backup_loss_func = torch.nn.BCELoss()
                self.loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
            else:
                self.loss_func = torch.nn.BCEWithLogitsLoss()

            self.model = self.model.float().to(dev)
            self.data_loader = DataLoader(
                self.dataset, batch_size=self.params["batch_size"], shuffle=True
            )

            self.opt = optim.Adam(self.model.parameters(), lr=self.params["lr"])
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt)

            hyperopt_metric = self.fit_torch_nn()
        return hyperopt_metric

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
                val_metric_values, val_sum_stat = self.run_epoch()

                self.metrics["val"].append(val_metric_values)
                val_loss = val_metric_values[loss_index]

            if self.trial:
                self.trial.report(val_sum_stat, step=self.epoch)

            # EARLY STOPPING CHECK
            stop_early = es_los.step(val_loss)

            # ******** TESTING ********
            if stop_early or self.epoch == self.params["epochs"] - 1:
                self.model.eval()
                self.dataset.status = "testing"
                with torch.no_grad():
                    self.metrics["test"], _ = self.run_epoch()
                return val_sum_stat

            self.scheduler.step(val_sum_stat)

    def run_epoch(self):
        total_loss = 0
        labels, predictions, probs = [], [], []

        for xb, yb in tqdm(self.data_loader):
            batch_loss, batch_predictions, batch_probs = self.run_batch(
                xb.to(dev), yb.to(dev)
            )
            total_loss += batch_loss

            labels.append(yb.numpy())
            probs.append(batch_probs.cpu().detach().numpy())
            predictions.append(batch_predictions.cpu().detach().numpy())

        avg_loss = total_loss / len(self.data_loader)

        report, summary_stat = self.calculate_metrics(
            np.vstack(labels), np.vstack(predictions), np.vstack(probs)
        )

        m_list, _ = self.listify_metrics(report, avg_loss)
        s = self.dataset.status
        print(f"{self.epoch}-{s}: L: {avg_loss:.3f} | Avg-F1 {summary_stat:.3f}")

        return m_list, summary_stat

    def run_batch(self, xb, yb):
        raw_out = self.model(xb)
        # BCEwithlogits does the sigmoid
        loss = self.loss_func(raw_out, yb)
        batch_loss = loss.item()

        # BCEwithlogits occassionally returns a huge loss
        # we replace these losses
        if batch_loss > 100:
            loss_alt = self.backup_loss_func(torch.sigmoid(raw_out), yb)
            batch_loss_alt = loss_alt.item()
            loss = loss_alt
            batch_loss = batch_loss_alt

        if self.dataset.status == "training":
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.opt.step()
            self.opt.zero_grad()

        probs = torch.sigmoid(raw_out)
        preds = torch.round(probs)

        # print("for input shape (b,w,f):", xb.shape)
        # print("labels were:", yb)
        # print("Model output", raw_out)
        # print("Probs were:", probs)
        # print("Loss of: ", batch_loss)
        return batch_loss, preds, probs

    def fit_sklearn_classifier(self):
        print(f"Fitting Sklearn Classifier - {self.params['model']}")
        X_train, X_val, X_test, Y_test, Y_train, Y_val = self.dataset.get_sk_dataset()
        sample_weights = compute_sample_weight(class_weight="balanced", y=Y_train)
        print("Fitting model")
        self.model.fit(X_train, Y_train, sample_weight=sample_weights)
        self.metrics = {}
        print("Testing fit")
        self.metrics["train"], _, _ = self.test_fit(X_train, Y_train)
        self.metrics["val"], _, val_result_summary = self.test_fit(X_val, Y_val)
        self.metrics["test"], test_results_df, test_result_summary = self.test_fit(
            X_test, Y_test
        )
        print(f"\nTest results:")
        print(test_results_df)

        return val_result_summary

    def test_fit(self, X_set, Y_set):
        Y_pred = self.model.predict(X_set)
        Y_prob = self.model.predict_proba(X_set)
        metrics_dict, m_summary = self.calculate_metrics(
            Y_set, Y_pred, probs=Y_prob, output_dict=True
        )
        m_list, m_df = self.listify_metrics(metrics_dict)
        return m_list, m_df, m_summary

    def calculate_metrics(
        self,
        labels,
        preds,
        probs=None,
        output_dict=True,
        summary_stat="macro avg",
        verbose=False,
    ):
        # Transform the metrics
        y_pred_list = [a.squeeze().tolist() for a in preds]
        y_labels = [a.squeeze().tolist() for a in labels]
        if type(y_labels[0]) is list:
            y_labels = [[int(a) for a in b] for b in y_labels]
        else:
            y_labels = [int(a) for a in y_labels]

        if probs is None:
            print("substituting probs")
            probs = y_pred_list
        else:
            probs = [a.squeeze().tolist() for a in probs]
            # probs = probs[0]

        # Calculate all the metrics
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        final_summary = []
        final_report = {}
        probs = np.array(probs)
        y_labels = np.array(y_labels)
        y_pred_list = np.array(y_pred_list)

        if self.params["model"] in ["xgb", "tree", "forest"]:
            # print(f"labels shape {y_labels.shape}")
            # print(f"preds shape {y_pred_list.shape}")
            # input(f"probs shape {probs.shape}")
            # probs = probs[:, :, 0]
            # probs = np.reshape(probs, y_labels.shape)
            # input(f"new probs shape {probs.shape}")
            probs = y_pred_list

        if len(y_labels.shape) == 1:
            probs = np.expand_dims(probs, 1)
            y_labels = np.expand_dims(y_labels, 1)
            y_pred_list = np.expand_dims(y_pred_list, 1)

        for i in range(len(self.params["class_names"])):
            try:
                sub_probs = probs[:, i].tolist()
                sub_y_labels = y_labels[:, i].tolist()
                sub_y_pred_list = y_pred_list[:, i].tolist()
            except IndexError as e:
                print(probs[:5])
                print(y_labels[:5])
                print(y_pred_list[:5])
                raise e

            tn, fp, fn, tp = confusion_matrix(sub_y_labels, sub_y_pred_list).ravel()

            try:
                auROC = roc_auc_score(sub_y_labels, sub_probs, average=None)
                AvgPrec = average_precision_score(sub_y_labels, sub_probs, average=None)
            except ValueError as V:
                print(self.params["class_names"])
                print("Probs:", probs[:6])
                print("Labels", y_labels[:6])
                print("Preds", y_pred_list[:6])
                print(
                    f"Label shape {len(sub_y_labels)}, probs shape {len(sub_probs)}, values used"
                )
                print(sub_y_labels[:10])
                print(sub_probs[:10])

                auROC = 0.510101
                AvgPrec = 0.10101
                raise V

            report = classification_report(
                sub_y_labels,
                sub_y_pred_list,
                output_dict=output_dict,
                # labels=[i for i in range(len(self.params["class_names"]))],
                labels=[1],
                target_names=[self.params["class_names"][i]],
            )
            if verbose:
                print("\n\n", self.params["class_names"][i])
                print(f"\nlabels: {sub_y_labels[:20]}")
                print(f"predictions: {sub_y_pred_list[:20]}\n")
                print(f"TP {tp}  TN {tn}   FP {fp}   FN {fn}")
                printable = [round(p, 2) for p in sub_probs[:20]]
                print(f"probs: {printable}")
                print(f"auROC: {auROC}, AP: {AvgPrec}")

            try:
                report[self.params["class_names"][i]]["auROC"] = auROC
                report[self.params["class_names"][i]]["AP"] = AvgPrec

                report[self.params["class_names"][i]]["conf-TP"] = tp
                report[self.params["class_names"][i]]["conf-TN"] = tn
                report[self.params["class_names"][i]]["conf-FP"] = fp
                report[self.params["class_names"][i]]["conf-FN"] = fn

            except Exception as e:
                print(report)
                raise e
            for k, v in report.items():
                if "avg" not in k:
                    print(k, v)
                    final_report[k] = v
            final_summary.append(report[summary_stat]["f1-score"])

        print(f"F1 by class {final_summary}")
        final_stat = sum(final_summary) / len(final_summary)
        print(f"Avg F1: {final_stat}")
        return final_report, final_stat

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

    def plot_metrics(self, verbose=False):
        for k in ["train", "val", "test"]:
            if k is "test" or self.params["model"] in [
                "forest",
                "tree",
                "mlp",
                "knn",
                "xgb",
            ]:
                df = pd.DataFrame([self.metrics[k]], columns=self.columns)
            else:
                df = pd.DataFrame(self.metrics[k], columns=self.columns)
            # print(k)
            # print(df)
            metrics_to_log = [
                "auROC",
                "AP",
                "support",
                "precision",
                "recall",
                "f1-score",
                "loss",
            ]
            for c in self.columns:
                for m in metrics_to_log:
                    if m in c:
                        if verbose:
                            print(f"{k}_{c}", df[c].iloc[-1])
                        elif "f1-score" in c:
                            print(f"{k}_{c}", df[c].iloc[-1])

        print("create graphs")
        if self.params["model"] in ["forest", "tree", "mlp", "knn", "xgb"]:
            df_train = pd.DataFrame([self.metrics["train"]], columns=self.columns)
            df_val = pd.DataFrame([self.metrics["val"]], columns=self.columns)
        else:
            df_train = pd.DataFrame(self.metrics["train"], columns=self.columns)
            df_val = pd.DataFrame(self.metrics["val"], columns=self.columns)

        print(df_val.shape)
        if df_val.shape[0] == 1:
            # Don't plot single point graphs
            return
        else:
            columns_to_plot = [
                c for c in df_train.columns if ("f1-score" in c or "AP" in c)
            ]
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
        return


if __name__ == "__main__":
    print("Starting")
    model_params = {}
    data_loader = LoadDF()
    df = data_loader.get_all_sessions()

    window = 5
    # classes = ["speaking", "finishing"]
    classes = ["finishing"]
    data = TimeSeriesDataset(df, window=window, overlap=False, labels=classes)

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
    model_params["class_names"] = classes
    model_params["num_classes"] = len(classes)
    model_params["num_features"] = data.df.shape[1]
    model_params["class_weights"] = data.weights

    trainer = ModelTraining(model_params, data, verbose=True)

    auc_roc = trainer.train_and_eval_model()
    trainer.plot_metrics()
