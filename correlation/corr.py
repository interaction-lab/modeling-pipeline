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


def corr(data, threshold=0.9):
    data = data.iloc[:, 1:-1]  # remove id and unnamed columns
    corr = data.corr()  #df.corr()
    sns.heatmap(corr)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Take a dataframe and return the features that are not correlated",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filename", help="path to input file")
    if(len(sys.argv) == 3):
        parser.add_argument("threshold", type=float, help="threshold to filter correlation")
        args = parser.parse_args()
        df = pd.read_csv(args.filename)
        print(corr(df, args.threshold))
    else:
        args = parser.parse_args()
        df = pd.read_csv(args.filename)
        print(corr(df))



