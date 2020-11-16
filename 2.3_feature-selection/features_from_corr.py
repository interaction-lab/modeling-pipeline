import argparse
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def calculateCorr(df, corr_method, threshold):
    """ Methods include 'pearson', 'kendall', 'spearman'
    """
    print("Calculating " + corr_method + "....")

    df = df.drop(
        columns=["frame", "confidence"], errors="ignore"
    )  # remove unrelated columns
    n = len(df.columns)
    corr = df.corr(method=corr_method)

    # compare the corr between features and remove one of them if corr >= 0.9
    columns = np.full((n,), True, dtype=bool)
    for i in range(n):
        if columns[i]:
            for j in range(i + 1, n):
                # compare with threshold and remove if larger
                if corr.iloc[i, j] >= threshold:
                    if columns[j]:
                        columns[j] = False
    selected_columns = df.columns[columns]
    df = df[selected_columns]
    return selected_columns


def intersection(feature_lists):
    return sorted(set.intersection(*[set(list) for list in feature_lists]))


def get_args() -> argparse.ArgumentParser:
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
        default="/media/chris/M2/2-Processed_Data/Video-OpenFace_win/5/center.csv",
    )

    parser.add_argument(
        "--threshold",
        help="threshold to filter correlation (higher yields more features)",
        dest="threshold",
        type=float,
        required=False,
        default=0.7,
    )

    parser.add_argument(
        "--corr_method",
        help="correlation calculation",
        dest="corr_method",
        type=str,
        required=False,
        default="pearson",
    )

    return parser.parse_args()


def main():
    args = get_args()

    df = pd.read_csv(args.input_filename)
    selected_features = calculateCorr(df, args.corr_method, args.threshold)
    for c in selected_features:
        print(c)
    # print(selected_features)


if __name__ == "__main__":
    main()
