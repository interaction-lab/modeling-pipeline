import numpy as np
import cv2 as cv
import os
import json
import numpy as np
import pandas as pd
import itertools
import argparse

def read_json(file):
    with open(file) as f:
        data = json.load(f)
    return data

def get_keypoints(data):
    '''
    Get the keypoints detected in the frame 
    'data' is the info read in by read_json(folder, frame), basically a json file loaded into useable form
    '''
    kp = np.zeros([len(data['people']), 75])
    if len(kp)==0:
        return np.ones([1,75]).reshape([1,-1])*np.nan
    for i in range(len(data['people'])):
        kp[i] = data['people'][i]['pose_keypoints_2d']
    confidences = kp[:,2::3].sum(axis=1)
    return kp[np.argmax(confidences), :].reshape([1,-1])

def main(json_path, csv_path):
    files = [f for f in os.listdir(json_path) if 'json' in f]
    df = pd.DataFrame()
    i=0
    for f in files:
        data = read_json(f'{json_path}/{f}')
        keypoints = pd.DataFrame(get_keypoints(data))
        df = df.append(keypoints, ignore_index=True)
        i=i+1
        if i%1000==0:
            df.to_csv(csv_path)    
    df.to_csv(csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', '--input', type = str, help='path to the json  with openpose data')
    parser.add_argument('-o', '--output', type = str, help='name of the csv to storing resulting dataframe')
    args = parser.parse_args()
    main(args.input, args.output)
