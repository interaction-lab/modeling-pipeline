import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
from scipy.stats import spearmanr

def calculateCorr(df, method, threshold):
    print("Calculating " + method + "....")

    df = df.drop(columns=['frame', 'confidence'])  # remove unrelated columns
    n = len(df.columns)
    corr = df
    if(method == "corr"):
        corr = corr.corr()  #df.corr()

    # compare the corr between features and remove one of them if corr >= 0.9
    columns = np.full((n,), True, dtype=bool)
    for i in range(n):
        if columns[i]:
            for j in range(i + 1, n):
                # calculate corr based on different corr methods
                val = 0
                if(method == "corr"):
                    val = corr.iloc[i, j]
                elif(method == "pearson"):
                    val, _ = pearsonr(df.iloc[:,i], df.iloc[:,j])
                else:
                    val, _ = spearmanr(df.iloc[:, i], df.iloc[:, j])

                # compare with threshold and remove if larger
                if val >= threshold:
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
        "--input_filename",
        help="path to input file",
        dest="input_filename",
        type=str,
        required=False,
        default="./10/center.csv"
    )

    parser.add_argument(
        "--threshold",
        help="threshold to filter correlation",
        dest="threshold",
        type=float,
        required=False,
        default=0.9
    )

    parser.add_argument(
        "--corr_method",
        help="correlation calculation",
        dest="corr_method",
        type=str,
        required=False,
        default="corr"
    )

    return parser


if __name__ == "__main__":
    parsedArgs = get_arg_parser().parse_args()

    df = pd.read_csv(parsedArgs.input_filename)
    selected_features = calculateCorr(df, parsedArgs.corr_method, parsedArgs.threshold)
    print(selected_features)

