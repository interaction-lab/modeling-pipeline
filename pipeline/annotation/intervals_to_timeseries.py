import argparse
import os
import json
import swifter
import pandas as pd
import numpy as np

def main(interval_file, output, frequency, delimiter):
    df = pd.read_csv(interval_file, delimiter=delimiter)[['start', 'end', 'label']]
    if ~(df['start'].dtype == np.float64): #if not already in seconds, convert to seconds
        df['start'] = df.start.str[:2].astype(np.float64)*60*60 + df.start.str[3:5].astype(np.float64)*60 + df.start.str[6:].astype(np.float64)
        df['end'] = df.end.str[:2].astype(np.float64)*60*60 + df.end.str[3:5].astype(np.float64)*60 + df.end.str[6:].astype(np.float64)
    # create a new dataframe formatted as a one-hot-encoded timeseries
    print(df)
    start_time, end_time = df.start[0], df.end.values[-1]
    time = np.arange(0, end_time, 1/frequency)
    labels = np.unique(df.label)
    timeseries = pd.DataFrame(time, columns=['seconds'])
    # add a column for the second, and a one-hot encoded column for each label
    for l in labels:
        timeseries[l] = 0 
    # add in the correct labels
    for i, row in df.iterrows():
        # note: timesteps that start before the first annotation will have 0 for all encodings
        timeseries.loc[(timeseries.seconds >= row.start) & (timeseries.seconds < row.end), row.label] = 1
    # save as a new csv where specified!
    timeseries.to_csv(output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Take .txt or .csv file with intervals and creates a csv of labels, with 1 row per timestep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--interval_file", help="file with start, end, and label columns (string with hh:mm:ss.sss, or float with seconds)")
    parser.add_argument("-o", "--output", help="path to output json")
    parser.add_argument("-f", "--frequency", type=float, default=30, help="number of rows to generate per second")
    parser.add_argument("-d", "--delimiter", default = ',', help="the delimiter used in interval_file (, ; \\t etc)")
    args = parser.parse_args()

    main(args.interval_file, args.output, args.frequency, args.delimiter)