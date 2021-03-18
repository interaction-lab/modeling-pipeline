import sys
from tqdm import tqdm
import pandas as pd
import numpy as np

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import torch
from torch.utils.data import DataLoader
from torch import optim

from sklearn.utils.class_weight import compute_sample_weight


### Model Imports ###
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
# from sklearn.linear_model import SGDClassifier
# from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from .model_defs import TCNModel, RNNModel, LSTMModel, GRUModel


from .model_monitoring import EarlyStopping
from .data_utils import TimeSeriesDataset, LoadDF, timeit
from .model_performance import ModelMetrics


# Needed to reproduce
torch.manual_seed(0)
np.random.seed(0)

print(f"Cuda? {torch.cuda.is_available()}")
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ModelTraining:
    """The ModelTraing class is intended as model and dataset agnostic trainer (currently supporting time series datasets)
    """
    def __init__(self, params, dataset, trial=None, verbose=True):
        self.trial = trial
        self.dataset = dataset
        self.verbose = verbose
        self.metrics = {"train": [], "val": [], "test": []}
        self.model = self._load_model_with_params(params)
        self.mm = ModelMetrics(params)


    def _load_model_with_params(self, params):
        """Create model object with specified parameters

        Args:
            params (dict): dictionary of parameters for training a model
        """
        self.params = params
        print(f"starting round with these params:")
        for k,v in params.items():
            print(f"\t{k}: {v}")

        if params["model"] == "tcn":
            model = TCNModel(
                num_channels=[params["num_features"]] * params["num_layers"],
                window=params["window"],
                kernel_size=params["kern_size"],
                num_classes=params["num_classes"],
                dropout=params["dropout"],
            )
        elif params["model"] == "rnn":
            model = RNNModel(
                hidden_size=params["kern_size"],
                input_size=params["num_features"],
                num_layers=params["num_layers"],
                num_classes=params["num_classes"],
                dropout=params["dropout"],
            )
        elif params["model"] == "lstm":
            model = LSTMModel(
                input_size=params["num_features"],
                hidden_size=params["kern_size"],
                num_layers=params["num_layers"],
                num_classes=params["num_classes"],
                dropout=params["dropout"],
            )
        elif params["model"] == "gru":
            model = GRUModel(
                input_size=params["num_features"],
                hidden_size=params["kern_size"],
                num_layers=params["num_layers"],
                num_classes=params["num_classes"],
                dropout=params["dropout"],
            )
        elif params["model"] == "tree":
            model = DecisionTreeClassifier(
                max_depth=params["max_depth"],
                criterion=params["criterion"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                class_weight="balanced",
            )
        elif params["model"] == "forest":
            model = RandomForestClassifier(
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
                model = MultiOutputClassifier(
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
                model = xgb.XGBClassifier(
                    objective="binary:logistic",
                    max_depth=params["max_depth"],
                    num_class=params["num_classes"],
                    booster=params["booster"],
                    learning_rate=params["learning_rate"],
                    verbosity=1,
                )
        elif params["model"] == "knn":
            model = KNeighborsClassifier(
                n_neighbors=params["n_neighbors"],
                leaf_size=params["leaf_size"],
                verbose=True,
            )
        elif params["model"] == "mlp":
            hidden_layer_sizes = tuple(
                [params["width"] // i for i in range(1, params["depth"] + 1)]
            )
            model = MLPClassifier(
                solver=params["solver"],
                activation=params["activation"],
                learning_rate="adaptive",
                learning_rate_init=params["learning_rate"],
                hidden_layer_sizes=hidden_layer_sizes,
                verbose=True,
            )
        return model

    @timeit
    def train_and_eval_model(self):
        """Setup and train model on the provided datset

        Will call training functions (fit_) for sklearn classifiers or torch 
        neural networks.

        Returns:
            float: performance of the metric being optimized
        """

        ### SKLearn Models
        if self.params["model"] in ["xgb", "tree", "forest", "knn", "mlp"]:
            hyperopt_metric = self.fit_sklearn_classifier()

        ### PyTorch Models
        else:
            hyperopt_metric = self.fit_torch_nn()
        return hyperopt_metric

    def fit_torch_nn(self):
        """Setup and train a NN with pytorch

        Includes dataset weighting, ADAM optimizer with learning rate scheduling,
        and early stopping to prevent overfitting.

        Each epoch will run on the training and validation sets, and the final run
        will include the test set. The final return value is the metrics performance
        on the validation set (used for hyperpameter tuning).

        Returns:
            float: performance metric on the validation set.
        """
        ### Setup Torch Model Training
        if self.params["weight_classes"]:
            weights = torch.FloatTensor(self.dataset.weights).float().to(dev)
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

        es_los = EarlyStopping("val_loss", patience=self.params["patience"])

        print(f"\nFitting PyTorch Classifier - *** {self.params['model']} *** ")
        for self.epoch in tqdm(range(self.params["epochs"])):

            # ******** TRAINING ********
            self.model.train()
            self.dataset.status = "training"

            train_metric_values, _ = self.run_epoch()

            self.metrics["train"].append(train_metric_values)
            loss_index = self.metrics_names.index("loss")
            # ******** TRAINING ********

            # ******** EVALUATION ********
            self.model.eval()
            self.dataset.status = "validation"

            with torch.no_grad():
                val_metric_values, val_sum_stat = self.run_epoch()

                self.metrics["val"].append(val_metric_values)
                val_loss = val_metric_values[loss_index]
            # ******** EVALUATION ********

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
            # ******** TESTING ********

            self.scheduler.step(val_sum_stat)

    def run_epoch(self):
        """Run the model on the dataset

        Includes backprop if dataset is 'training'

        Will take model outputs and calculate all performance metrics.

        Returns:
            list: all the metrics organized as a list
            float: the summary statistic to keep
        """
        total_loss = 0
        labels, predictions, probs = [], [], []

        ### Run through all the batches
        for xb, yb in tqdm(self.data_loader):
            batch_loss, batch_predictions, batch_probs = self.run_batch(
                xb.to(dev), yb.to(dev)
            )
            total_loss += batch_loss

            labels.append(yb.numpy())
            probs.append(batch_probs.cpu().detach().numpy())
            predictions.append(batch_predictions.cpu().detach().numpy())

        ### Calculate the loss for this epoch
        avg_loss = total_loss / len(self.data_loader)

        ### Calculate all metrics of interest
        report, summary_stat = self.mm.calculate_metrics(
            np.vstack(labels), np.vstack(predictions), np.vstack(probs)
        )

        m_list, self.metrics_names = self.mm.listify_metrics(report, avg_loss)

        print(f"{self.epoch}-{self.dataset.status}: L: {avg_loss:.3f} | Avg-F1 {summary_stat:.3f}")

        return m_list, summary_stat

    def run_batch(self, xb, yb, debug=False):
        """Run model on batch input, backprop if training

        Args:
            xb (tensor): model input
            yb (tensor): labels
            debug (bool, optional): print full input and output of the model. Defaults to False.

        Returns:
            float: loss for this batch
            tensor: model predictions
            tensor: model output probabilities
        """

        raw_out = self.model(xb)
        # BCEwithlogits does the sigmoid
        loss = self.loss_func(raw_out, yb)
        batch_loss = loss.item()

        # BCEwithlogits occassionally returns a huge loss - we replace these losses
        # TODO: debug why this happens and how to avoid
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

        if debug:
            print("for input shape (b,w,f):", xb.shape)
            print("labels were:", yb)
            print("Model output", raw_out)
            print("Probs were:", probs)
            print("Loss of: ", batch_loss)
            input("Hit enter to continue")
        return batch_loss, preds, probs

    def fit_sklearn_classifier(self):
        """Fits the sklearn classifier to the training data and evaluates it.

        Sample weights are computed on the validation and testing sets.

        Returns:
            float: performance metric on the validation set.
        """
        print(f"Fitting Sklearn Classifier -  *** {self.params['model']} *** ")

        # SKlearn trains on the whole dataset at once so it is all loaded
        # TODO: for large datasets, implement training pipeline with batches

        self.dataset.status = "training"
        X, Y = self.dataset.get_sk_dataset()

        # Training weights are adjusted to account for class imbalance
        sample_weights = compute_sample_weight(class_weight="balanced", y=Y)

        print("Fitting model")
        self.model.fit(X, Y, sample_weight=sample_weights)

        print("Testing fit")
        self.metrics["train"], _ = self.test_fit(X, Y)

        self.dataset.status = "validation"
        X, Y = self.dataset.get_sk_dataset()
        self.metrics["val"], val_result_summary = self.test_fit(X, Y)

        self.dataset.status = "validation"
        X, Y = self.dataset.get_sk_dataset()
        self.metrics["test"], test_result_summary = self.test_fit(X,Y)

        print(f"\nTest results:")
        test_results_df = pd.DataFrame([self.metrics["test"]], columns=self.metrics_names)
        print(test_results_df)

        return val_result_summary

    def test_fit(self, X_set, Y_set):
        """Evaluate performance of sklearn model on dataset

        Produces predictions and probabilities on the model input for performance calculation

        Args:
            X_set (ndarray): 2D model input where each row is an example
            Y_set (ndarray): labels for each input

        Returns:
            performance stats: performance stats as a list, a dataframe, and a single value
        """
        Y_pred = self.model.predict(X_set)
        Y_prob = self.model.predict_proba(X_set)
        metrics_dict, m_summary = self.mm.calculate_metrics(
            Y_set, Y_pred, probs=Y_prob, output_dict=True
        )
        m_list, self.metrics_names = self.mm.listify_metrics(metrics_dict)
        return m_list, m_summary




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
    trainer.mm.plot_metrics()
