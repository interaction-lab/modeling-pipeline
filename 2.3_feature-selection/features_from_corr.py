import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings

warnings.filterwarnings("ignore")
import statsmodels.api as sm
from scipy.stats import pearsonr
from scipy.stats import spearmanr


def corr(data, threshold):
    data = data.iloc[:, 1:-1]  # remove id and unnamed columns
    corr = data.corr()  # df.corr()
    # sns.heatmap(corr)

    # compare the corr between features and remove one of them if corr >= 0.9
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        if columns[i]:
            for j in range(i + 1, corr.shape[0]):
                if corr.iloc[i, j] >= threshold:
                    if columns[j]:
                        columns[j] = False
    selected_columns = data.columns[columns]
    data = data[selected_columns]
    return selected_columns


def pearson(df, threshold):
    df = df.iloc[:, 1:-1]  # remove id and unnamed columns
    n = len(df.columns)
    # compare the pearson's correlation between features and remove one of them if corr >= 0.9
    columns = np.full((n,), True, dtype=bool)
    for i in range(n):
        if columns[i]:
            for j in range(i + 1, n):
                corr, _ = pearsonr(df.iloc[:, i], df.iloc[:, j])
                if corr >= threshold:
                    if columns[j]:
                        columns[j] = False
    selected_columns = df.columns[columns]
    df = df[selected_columns]
    return selected_columns


def spearman(df, threshold):
    df = df.iloc[:, 1:-1]  # remove id and unnamed columns
    n = len(df.columns)
    # compare the spearman's correlation between features and remove one of them if corr >= 0.9
    columns = np.full((n,), True, dtype=bool)
    for i in range(n):
        if columns[i]:
            for j in range(i + 1, n):
                corr, _ = spearmanr(df.iloc[:, i], df.iloc[:, j])
                if corr >= threshold:
                    if columns[j]:
                        columns[j] = False
    selected_columns = df.columns[columns]
    df = df[selected_columns]
    return selected_columns


def intersection(feature_lists):
    return sorted(set.intersection(*[set(list) for list in feature_lists]))


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Take a dataframe and return the features that are not correlated",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input_dir",
        help="directory of the input csv files",
        dest="input_dir",
        type=str,
        required=False,
        default="./10/",
    )

    parser.add_argument(
        "--input_filename",
        help="path to input file",
        dest="input_filename",
        type=str,
        required=False,
        default="center.csv",
    )

    parser.add_argument(
        "--threshold",
        help="threshold to filter correlation",
        dest="threshold",
        type=float,
        required=False,
        default=0.9,
    )

    parser.add_argument(
        "--corr_method",
        help="correlation calculation",
        dest="corr_method",
        type=str,
        required=False,
        default="corr",
    )

    return parser


if __name__ == "__main__":
    parsedArgs = get_arg_parser().parse_args()

    filename = parsedArgs.input_dir + parsedArgs.input_filename
    df = pd.read_csv(filename)
    selected_features = []
    if parsedArgs.corr_method == "corr":
        print("Calculating corr....")
        selected_features = corr(df, parsedArgs.threshold)
    elif parsedArgs.corr_method == "pearson":
        print("Calculating pearson...")
        selected_features = pearson(df, parsedArgs.threshold)
    else:
        print("Calculating spearman...")
        selected_features = spearman(df, parsedArgs.threshold)

    # selected_features1 = corr(df, parsedArgs.threshold)
    # selected_features2 = pearson(df, parsedArgs.threshold)
    # selected_features = intersection([selected_features1, selected_features2])
    print(selected_features)

