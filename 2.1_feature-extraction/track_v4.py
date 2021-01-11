import numpy as np
import cv2 as cv
import os
import json
import numpy as np
import pandas as pd
import itertools
import argparse

''' Create clusters for max 3 people based on the last 60 viable frames, and choose which person goes to which cluster '''
''' Update from last version: If there is a 4th person, choose which 3 people are best (mitigating problematic 4th people walking into the room)'''

def read_json(folder, frame):
    files = os.listdir(folder)
    with open(f'{folder}/{files[frame]}') as f:
        data = json.load(f)
    return data

def get_point(keyframe, keypoint):
    '''
    Given 'data' (extracted json data from one video frame), a specified person (int), and a keypoint (0 for nose, 1 for neck, etc)
    return the x and y coordinate
    '''
    x = keyframe[keypoint*3]
    y = keyframe[keypoint*3+1]
    return (int(x), int(y))

def get_keypoints(data):
    '''
    Get the keypoints detected in the frame 
    'data' is the info read in by read_json(folder, frame), basically a json file loaded into useable form
    '''
    kp = np.zeros([len(data['people']), 75])
    for i in range(len(data['people'])):
        kp[i] = data['people'][i]['pose_keypoints_2d']
    return kp

def get_keyframe_difference(keyframe1, keyframe2):
    '''
    Measure the mean absolute-valued difference between non-0 values of two equal-length numpy arrays
    This is used to compare between non-0 keypoints of detected people across frames
    '''
    keyframe1 = np.matrix(keyframe1).reshape(1,-1); keyframe2 = np.matrix(keyframe2).reshape(1,-1)
    x1, y1 = keyframe1[0,[0,3]],  keyframe1[0,[1,4]]
    x2, y2 = keyframe2[0,[0,3]],  keyframe2[0,[1,4]]
    tempx1 = x1; tempx2 = x2
    x1 = x1[(tempx1>1)&(tempx2>1)]
    x2 = x2[(tempx1>1)&(tempx2>1)]
    y1 = y1[(tempx1>1)&(tempx2>1)]
    y2 = y2[(tempx1>1)&(tempx2>1)]

    dist = np.sqrt(np.square(x2-x1))
    dist = np.mean(np.sqrt(dist))
    return np.float(dist)

def get_keypoint_center(keyframe):
    values = np.matrix(keyframe.reshape(1,-1))
    x = values[0,::3]
    y = values[0,1::3]
    temp = x
    x = x[temp>0] # only keep keyframes that are actually found (otherwise they are at 0)
    y = y[temp>0]
    
    xc = np.mean(x)
    yc = np.mean(y)
    c = {}
    c['x'] = xc
    c['y'] = yc
    return c

def get_cluster_center(cluster):
    x = np.mean(cluster['x'])
    y = np.mean(cluster['y'])
    c = {}
    c['x'] = x
    c['y'] = y
    return c

def center_distance(c1, c2):
    dist = np.sqrt(np.square(c2['x'] - c1['x']))
    return dist

def repeat_row(df):
    '''add a copy of the last row to the dataframe'''
    row = df.iloc[-1,:]
    df = df.append(row, ignore_index=True)
    return df

def sort_people(cl, df, data_curr, n):
    '''
    cl: the array of clusters
    df: the array of dataframes
    data_curr: the extracted json data of the current frame
    '''
    # get the keypoints of the current frame
    kp_curr = get_keypoints(data_curr)
    n_people = len(kp_curr)

    # get the center of each cluster
    centers = []
    for i in range(n):
        centers.append(get_cluster_center(cl[i]))

    # if there are at least n people detected:
    if n_people < 1:
        for i in range(n): # just repeat the last row of these dataframes- their corresponding person wasn't detected in these frames
            df[i] = repeat_row(df[i])
        return cl, df
    elif n_people >= n:
        # each possibility for matching detected people to clusters,
        # where i is the cluster and o[i] is number of the detected person/keyframe (ordered by openpose)
        options = itertools.permutations(range(n_people))
        min_dist = 100000000000
        best_option = None
        for o in options: # for each possible combination of clusters and keypoints, calculate the total distance between centers
            dist = 0
            # calculate the mean distance between clusters and next keypoints
            for i in range(n):
                dist = dist + center_distance(centers[i], get_keypoint_center(kp_curr[o[i]]))
            if dist < min_dist:
                min_dist = dist; best_option = o
        # once we decide the best option for which keypoints relate to which cluster, we update the clusters and the dataframes!
        for i in range(n):
            # update dataframe i
            keyframe = kp_curr[best_option[i]]
            keyframe_df = pd.DataFrame(np.matrix(keyframe).reshape(1,-1))
            df[i] = df[i].append(keyframe_df)
            # update cluster
            cl[i] = update_cluster(cl[i], get_keypoint_center(keyframe))
    else:
        options = itertools.permutations(range(n))
        min_dist = 100000000000
        best_option = None
        # o[i] is the cluster, i is the number of the detected person/keyframe
        for o in options: # for each possible combination of clusters and keypoints, calculate the total distance between centers
            dist = 0
            # calculate the mean distance between clusters and next keypoints
            for i in range(n_people):
                dist = dist + center_distance(centers[o[i]], get_keypoint_center(kp_curr[i]))
            if dist < min_dist:
                min_dist = dist; best_option = o
        for i in range(n_people): # update the clusters where people were found
            # update dataframe i
            keyframe = kp_curr[i]
            keyframe_df = pd.DataFrame(np.matrix(keyframe).reshape(1,-1))
            df[best_option[i]] = df[best_option[i]].append(keyframe_df)
            # update cluster
            cl[best_option[i]] = update_cluster(cl[best_option[i]], get_keypoint_center(keyframe))
        for i in range(n_people, n): # just repeat the last row of these dataframes- their corresponding person wasn't detected in these frames
            df[best_option[i]] = repeat_row(df[best_option[i]])

    return cl, df

def create_cluster(x,y):
    cluster = {}
    cluster['x'] = [x]
    cluster['y'] = [y]
    return cluster

def update_cluster(cluster, keyframe_center):
    max_length = 120
    x, y = keyframe_center['x'], keyframe_center['y']
    if len(cluster['x']) < max_length:
        cluster['x'].append(x)
        cluster['y'].append(y)
    else:
        cluster['x'].pop(0)
        cluster['x'].append(x)
        cluster['y'].pop(0)
        cluster['y'].append(y)
    return cluster

def main(n, input_video, json_path, output_folder):
    
    ################################### INITIALIZATION #######################################
    # initializing variables for video, if needed
    if input_video != '0': 
        cap = cv.VideoCapture(input_video)
    font = cv.FONT_HERSHEY_SIMPLEX

    # create one dataframe and one cluster object per person, and store them in an array
    df = []
    cl = []
    for i in range(n):
        df.append(pd.DataFrame())
        cl.append(None)
 
    # 'initialize' each cluster using the first videoframe. 
    # in other words, assign the first n people detected each to one cluster
    # Here I do assume that the most prominent people as detected by openpose in the first frame are the people you are trying to track, and that there are at least n people detected by openpose in this frame
    data = read_json(json_path, 0)
    kp = get_keypoints(data)
    for i in range(n):
        kp_i = kp[i]
        c_i = get_keypoint_center(kp_i)
        cl[i] = create_cluster(c_i['x'], c_i['y'])

    cl, df = sort_people(cl, df, data, n)

    # If specified, read the first frame of the video
    if input_video != '0': 
        ret, frame2 = cap.read()

    # ################################### LOOP THROUGH FRAMES ###################################
    i=1
    length = len(os.listdir(json_path))
    while(i<length):
        # 1. get current frame data from the json file
        if input_video != '0': 
            ret, frame2 = cap.read()

        # 2. read in the data for the current frames
        data_curr = read_json(json_path, i)

        # 2. Determine which keypoints are person 1, person 2, person 3 and update dataframes as needed
        cl, df = sort_people(cl, df, data_curr, n)
        # 3. If specified, draw numbers on the people in the video
        if input_video != '0':
            try:
                keypoints = get_keypoints(data_curr)
                for j in range(n):
                    cv.putText(frame2, f'p{j}', ((int(cl[j]['x'][-1]), int(cl[j]['y'][-1]))), font, 1, (0,0,255), 2)
            except IndexError:
                point = [100,100]
                text = 'None Detected'
            cv.imshow('frame2', frame2)  

        # 4. Increment the counter
        i=i+1

        # Enable quitting
        k = cv.waitKey(30) & 0xff
        if k == 27:
            sorted_df.to_csv(f'{output_folder}/sorted_keyframes.csv')
            break
        elif k == ord('s'):
            continue 

    # Save the results to a csv    
    for i in range(n):
        df[i].to_csv(f'{output_folder}/person{i}.csv')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-n', '--num_people', type=int, help='number of people to detect')
    parser.add_argument('-i', '--input_video', type = str, help='path to the input video')
    parser.add_argument('-j', '--json_path', type = str, help='path to the json directory')
    parser.add_argument('-o', '--output_folder', type = str, help='folder for storing results')
    args = parser.parse_args()
    main(args.num_people, args.input_video, args.json_path, args.output_folder)


    