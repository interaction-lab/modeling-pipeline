import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    plot_confusion_matrix,
    average_precision_score,
)


class ModelMetrics:

    def __init__(self, params) -> None:
        self.params = params

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
        if verbose:
            self.graph_model_output(
                y_labels, y_pred_list, probabilities=probs, title="title"
            )
        return final_report, final_stat

    def listify_metrics(self, metrics_dict, loss=0):
        """Convert metrics from a dictionary to a list

        The dictionary of all metrics is converted to a list and the columns 
        are saved in self.metrics_names. The loss is not always used for 

        Args:
            metrics_dict (dict): dictionary of all performance metrics
            loss (int, optional): cumulative loss for a given epoch. Defaults to 0.

        Returns:
            list, DataFrame: a list of the performance metrics values and a dataframe
        """
        # if loss:
        #     columns = ["loss"]
        #     values = [loss]
        # else:
        #     columns, values = [], []
        columns = ["loss"]
        values = [loss]
        for k, v in metrics_dict.items():
            for k2, v2 in v.items():
                columns.append(f"{k}-{k2}")
                values.append(v2)
        
        self.metrics_names = columns
        return values, columns


    def graph_model_output(
        self,
        actual_labels,
        predicted_labels,
        probabilities=None,
        max_graph_size=1000,
        title="Graph Title",
    ):
        print("Graphing model output")
        # TODO: Add saving of outputs
        assert (
            actual_labels is not None and predicted_labels is not None
        ), "Invalid inputs to graph model, must not be none!"

        input_length = len(actual_labels)
        if input_length > max_graph_size:
            for i in range(math.ceil(input_length / max_graph_size)):
                plt.figure(figsize=(30, 5))
                plt.plot(
                    actual_labels[max_graph_size * i : max_graph_size * (i + 1)],
                    color="green",
                    label="original",
                )
                plt.plot(
                    predicted_labels[max_graph_size * i : max_graph_size * (i + 1)]
                    * 0.9,
                    color="red",
                    label="predicted",
                )  # times 0.9 to differentiate lines
                if probabilities is not None:
                    plt.plot(
                        probabilities[max_graph_size * i : max_graph_size * (i + 1)],
                        color="yellow",
                        label="probability",
                    )

                plt.legend()
                plt.set_title(title)
                plt.show()
        else:
            plt.figure(figsize=(30, 5))
            plt.plot(actual_labels, color="green", label="original")
            plt.plot(
                predicted_labels * 0.9, color="red", label="predicted"
            )  # times 0.9 to differentiate lines
            if probabilities is not None:
                plt.plot(probabilities, color="yellow", label="probability")

            plt.legend()
            plt.show()

    def plot_metrics(self, metrics, metrics_names, verbose=False):
        for k in ["train", "val", "test"]:
            if k is "test" or self.params["model"] in [
                "forest",
                "tree",
                "mlp",
                "knn",
                "xgb",
            ]:
                df = pd.DataFrame([metrics[k]], columns=metrics_names)
            else:
                df = pd.DataFrame(metrics[k], columns=metrics_names)
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
            for c in metrics_names:
                for m in metrics_to_log:
                    if m in c:
                        if verbose:
                            print(f"{k}_{c}", df[c].iloc[-1])
                        elif "f1-score" in c:
                            print(f"{k}_{c}", df[c].iloc[-1])

        if self.params["model"] in ["forest", "tree", "mlp", "knn", "xgb"]:
            df_train = pd.DataFrame([metrics["train"]], columns=metrics_names)
            df_val = pd.DataFrame([metrics["val"]], columns=metrics_names)
        else:
            df_train = pd.DataFrame(metrics["train"], columns=metrics_names)
            df_val = pd.DataFrame(metrics["val"], columns=metrics_names)

        print(df_val.shape)
        if df_val.shape[0] == 1:
            print("No graphs to plot for trainers without epochs")
            # Don't plot single point graphs
            return
        else:
            columns_to_plot = [
                c for c in df_train.columns if ("f1-score" in c or "AP" in c)
            ]
            # Create a figure showing metrics progress while training
            _, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))
            df_train[columns_to_plot].plot(ax=axes[0])
            df_train["loss"].plot(ax=axes[0], secondary_y=True, color="black")
            axes[0].set_title("train")

            df_val[columns_to_plot].plot(ax=axes[1])
            df_val["loss"].plot(ax=axes[1], secondary_y=True, color="black")
            axes[1].set_title("val")

            plt.show()
            # experiment.log_image("diagrams", fig)
        return

# def get_all_metrics(turn_labels, probs, preds, verbose, plot):
#     auROC=roc_auc_score(turn_labels, probs)
#     AP=average_precision_score(turn_labels, probs)
#     fpr, tpr, thresholds = roc_curve(turn_labels, probs)

#     report = classification_report( turn_labels, preds, labels=[0,1], output_dict=True)
#     tn, fp, fn, tp = confusion_matrix(turn_labels, preds).ravel()
#     optimal_idx = np.argmax(tpr - fpr)
#     optimal_threshold = thresholds[optimal_idx]

#     # Adjusting for optimal threshold
#     adjust = .5-optimal_threshold
#     # adjust = optimal_threshold-.5
#     probs2 = probs + (adjust)
#     preds2 = probs2.round()#.fillna(0).reset_index(drop=True)
#     if plot:
#         fig, ax = plt.subplots(figsize=(20,3))
#         ax.plot(probs[:2000], 'r')
#         ax.plot(probs2[:2000], 'g')
#         plt.show()
#         fig, ax = plt.subplots(figsize=(20,3))
#         ax.plot(preds[:2000], 'r')
#         ax.plot(preds2[:2000], 'g')
#         plt.show()

#     auROC2=roc_auc_score(turn_labels, probs2)
#     AP2=average_precision_score(turn_labels, probs2)
#     fpr2, tpr2, thresholds2 = roc_curve(turn_labels, probs2)

#     report2 = classification_report( turn_labels, preds2, labels=[0,1], output_dict=True)
#     tn2, fp2, fn2, tp2 = confusion_matrix(turn_labels, preds2).ravel()
#     optimal_idx2 = np.argmax(tpr2 - fpr2)
#     optimal_threshold2 = thresholds2[optimal_idx2]

#     if verbose:
#         # print(f"\nFor session {session} person {person}:")
#         print("\nBEFORE")
#         print(f"auROC={auROC}")
#         print(f"AP={AP}")
#         print(f'Acc={report["accuracy"]}')
#         print(f'F1={report["weighted avg"]["f1-score"]}')
#         print(f"\n Optimal threshold value is: {optimal_threshold}\n")
#         for k,v in report.items():
#             print(k,v)
#         print(f"tp={tp}, tn={tn}, fp={fp}, fn={fn}")

#         print("\nAFTER")
#         print(f"adjusting by {adjust}")
#         print(f"auROC={auROC2}")
#         print(f"AP={AP2}")
#         print(f'Acc={report2["accuracy"]}')
#         print(f'F1={report2["weighted avg"]["f1-score"]}')
#         print(f"\n Optimal threshold value is: {optimal_threshold2}\n")
#         for k,v in report2.items():
#             print(k,v)
#         print(f"tp={tp2}, tn={tn2}, fp={fp2}, fn={fn2}")

#     return auROC, AP, report, optimal_threshold, report2