__author__ = 'Leandra'

import pandas as pd
import os
import numpy as np
import math
from pyAudioAnalysis import audioAnalysis

def load_df(path_name, file_name):
    file_path = os.path.join(path_name, file_name)
    df = pd.read_csv(file_path)
    return df

def xy_centroid_to_num(x, y):
    scale_factor = 10.0 #3840 dictionary size
    x_max = 800
    y_max = 480
    new_y = math.floor(y/scale_factor)
    if new_y <= 0:
        new_y = 1
    if new_y > y_max:
        new_y = y_max - 1
    new_x = math.floor(x/scale_factor)
    if new_x <= 0:
        new_x = 1
    if new_x > x_max:
        new_x = x_max - 1
    emb = (new_y - 1) + new_x*(y_max/scale_factor)
    emb = int(emb)
    return emb

def get_pos1(path_name, file_name):
    #remove last 3 characters
    file_name = file_name[:-3]
    path_name = os.path.join(path_name, "audio")
    file_path = os.path.join(path_name, file_name + "wav")
    pos1 = audioAnalysis.beatExtractionWrapperBoolean(file_path)
    return pos1

def split_seq(arr, chunk_size):
    chunks = [arr[x:x+chunk_size] for x in xrange(0, len(arr) - (len(arr) % chunk_size), chunk_size)]
    return chunks

def input_output_seq(file_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = "song_dance_raw_data"
    dir_path = os.path.join(dir_path, "..") #one directory up from this file
    dir_path = os.path.join(dir_path, data_dir)
    unique_embedding_vals = set()
    input_seq_list = []
    output_seq_list = []
    sequence_length = 200
    print(file_name)
    df = load_df(dir_path, file_name)
    df = df.fillna(0)
    unique_timestamps = df.timestamp.unique()
    pos1 = get_pos1(dir_path,file_name)
    pos1 = pos1*50
    sorted_idx = np.searchsorted(unique_timestamps, pos1)
    sorted_idx[sorted_idx >= len(unique_timestamps)] = len(unique_timestamps) - 1
    pos1_aligned = [unique_timestamps[i] for i in sorted_idx]
    input_seq_x = []
    input_seq_y = []
    output_seq = []
    for timestamp in unique_timestamps:
        joints = df.loc[df['timestamp'] == timestamp]
        if (len(joints)) == 20: #valid time stamp - have all joint information
            #find centroid of points
            x_centroid = joints[" depth_x"].mean()
            y_centroid = joints[" depth_y"].mean()
            embedding = xy_centroid_to_num(x_centroid, y_centroid)
            #print (str(x_centroid) + ",  " + str(y_centroid))
            #print(embedding)
            if timestamp in pos1_aligned:
                output_seq.append(1) #beat
            else:
                output_seq.append(0) #no beat
            input_seq_x.append(x_centroid)
            input_seq_y.append(y_centroid)

    #print(input_seq) #centroid tuples
    #print(output_seq)
    output_np = np.array(output_seq)
    input_np_x = np.array(input_seq_x)
    input_np_y = np.array(input_seq_y)
    print('\n >> Save numpy arrays to disk')
    # save them
    np.save('input_seq_x.npy', input_np_x)
    np.save('input_seq_y.npy', input_np_y)
    np.save('output_seq.npy', output_np)
    #return input_seq, output_seq


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = "song_dance_raw_data"
    dir_path = os.path.join(dir_path, "..") #one directory up from this file
    dir_path = os.path.join(dir_path, data_dir)
    unique_embedding_vals = set()
    input_seq_list = []
    output_seq_list = []
    sequence_length = 200
    for file_name in os.listdir(dir_path):
        if ".csv" in file_name:
            print(file_name)
            df = load_df(dir_path, file_name)
            df = df.fillna(0)
            unique_timestamps = df.timestamp.unique()
            pos1 = get_pos1(dir_path,file_name)
            pos1 = pos1*50
            sorted_idx = np.searchsorted(unique_timestamps, pos1)
            sorted_idx[sorted_idx >= len(unique_timestamps)] = len(unique_timestamps) - 1
            pos1_aligned = [unique_timestamps[i] for i in sorted_idx]
            input_seq = []
            output_seq = []
            for timestamp in unique_timestamps[:int(len(unique_timestamps)*.2)]:
                joints = df.loc[df['timestamp'] == timestamp]
                if (len(joints)) == 20: #valid time stamp - have all joint information
                    #find centroid of points
                    x_centroid = joints[" depth_x"].mean()
                    y_centroid = joints[" depth_y"].mean()
                    embedding = xy_centroid_to_num(x_centroid, y_centroid)
                    #print (str(x_centroid) + ",  " + str(y_centroid))
                    #print(embedding)
                    if timestamp in pos1_aligned:
                        output_seq.append(1) #beat
                    else:
                        output_seq.append(0) #no beat
                    input_seq.append(embedding)

            print(input_seq)
            print(output_seq)
            input_seq_list.extend(split_seq(input_seq, sequence_length))
            output_seq_list.extend(split_seq(output_seq, sequence_length))
    output_np = np.array(output_seq_list)
    input_np = np.array(input_seq_list)
    print('\n >> Save numpy arrays to disk')
    # save them
    np.save('idx_q.npy', input_np)
    np.save('idx_a.npy', output_np)


if __name__ == "__main__":
   input_output_seq("kinect_skeleton04_23_17_13_24.csv")






