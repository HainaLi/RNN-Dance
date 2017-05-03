from __future__ import print_function
__author__ = 'Leandra'


import os, glob
import numpy as np
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

X_MAX = 640
Y_MAX = 480

def minmax_convert_file(input_filepath, output_folderpath):
    output_filepath = os.path.join(output_folderpath, 'minmaxnormalized_' + filename)
    # open csv file
    #input_file = open(input_filepath, 'rb')
    #input_reader = csv.reader(input_file)
    #header = next(input_reader)
    #output_file = open(output_filepath, "w")

    #print(header, file=output_file)

    dataframe = pd.read_csv(input_filepath, sep=r'\s*,\s*', engine='python')
    print(dataframe)
    # also scale numerical columns to range [0, 1]
    #for column in ['depth_x','depth_y']:
        #dataframe[column] = (dataframe[column] - dataframe[column].min()) / \
        #                        (dataframe[column].max() - dataframe[column].min())
    dataframe['depth_x'] = dataframe['depth_x'] / X_MAX
    dataframe['depth_y'] = dataframe['depth_y'] / Y_MAX

    dataframe.to_csv(output_filepath)
    return 0

def get_mean_std(input_folderpath):
    allFiles = glob.glob(input_folderpath + "/*.csv")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_,index_col=None, header=0)
        list_.append(df)
    frame = pd.concat(list_)
    mean = {}
    stdev = {}
    for column in frame.columns:
        mean[column.strip()] = frame[column].mean()
        stdev[column.strip()] = frame[column].std()
    return mean, stdev

def meanstd_convert_file(mean, stdev, input_filepath, output_folderpath):
    output_filepath = os.path.join(output_folderpath, 'meanstdnormalized_' + filename)
    # open csv file
    #input_file = open(input_filepath, 'rb')
    #input_reader = csv.reader(input_file)
    #header = next(input_reader)
    #output_file = open(output_filepath, "w")

    #print(header, file=output_file)

    dataframe = pd.read_csv(input_filepath, sep=r'\s*,\s*', engine='python')

    # also scale numerical columns to range [0, 1]
    for column in ['x','y','z','depth_x','depth_y']:
        dataframe[column] = (dataframe[column] - mean[column])/stdev[column]

    dataframe.to_csv(output_filepath)
    return 0



if __name__ == '__main__':
    normalize_method = "minmax"
    if normalize_method == "minmax":
        input_folder_name = 'to_normalize'
        output_folder_name = 'normalized'
        current_dir = os.path.dirname(os.path.realpath(__file__))
        to_normalize_path = os.path.join(current_dir, input_folder_name)
        output_folderpath = os.path.join(current_dir, output_folder_name)
        for filename in os.listdir(to_normalize_path):
            filepath = os.path.join(to_normalize_path, filename)
            print('normalized_' + filename)
            minmax_convert_file(filepath, output_folderpath)
    else:
        input_folder_name = 'to_normalize'
        output_folder_name = 'normalized'
        current_dir = os.path.dirname(os.path.realpath(__file__))
        to_normalize_path = os.path.join(current_dir, input_folder_name)
        output_folderpath = os.path.join(current_dir, output_folder_name)
        mean, stdev = get_mean_std(to_normalize_path)
        print (mean)
        for filename in os.listdir(to_normalize_path):
            filepath = os.path.join(to_normalize_path, filename)
            print('normalized_' + filename)
            #minmax_convert_file(filepath, output_folderpath)
            meanstd_convert_file(mean, stdev, filepath, output_folderpath)
