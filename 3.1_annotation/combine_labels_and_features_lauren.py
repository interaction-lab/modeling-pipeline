import argparse
import numpy as np
import pandas as pd

def main(label_file, feature_file, num_labels, output):
    label_df = pd.read_csv(label_file)
    # note: for my data only, as some infants may not have "f" or "c" instances, I need to add these in
    cols = label_df.columns
    if 'f' not in cols:
        label_df['f']=0
    if 'c' not in cols: 
        label_df['c']=0
    # end of part that is for my data only
    feature_df = pd.read_csv(feature_file)
    length = label_df.shape[0]
    feature_df = feature_df.iloc[:length, :]
    combined = pd.concat([feature_df, label_df.iloc[:, -num_labels:]], axis=1, ignore_index=True)
    combined = combined.loc[np.sum(combined.iloc[:, -num_labels:].values, axis=1)==1]
    columns = np.append(arr = feature_df.columns.values, values = ['a', 'f', 'c']) # this is also for my data only
    combined.columns = columns
    combined.to_csv(output, index=False)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Take .txt or .csv file with intervals and creates a csv of labels, with 1 row per timestep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-l", "--label_file", help="file with labels (one row per timestamp)")
    parser.add_argument("-f", "--feature_file", help="file with features (one row per timestamp)")
    parser.add_argument("-n", "--num_labels", type=int, help="number of labels in the label spreadsheet- assuming the labels are the last columns)")
    parser.add_argument("-o", "--output", help="path to output json")
    args = parser.parse_args()
    main(args.label_file, args.feature_file, args.num_labels, args.output)