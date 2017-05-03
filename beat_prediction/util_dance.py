import numpy as np
import math
import random
import os
import xml.etree.ElementTree as ET
import csv
import sys


class DataLoader():
    def __init__(self, limit = 500):
        self.data_dir = './dance_data'
        self.batch_size = 189
        self.tsteps = 100
        self.limit = limit # removes large noisy gaps in the data
        self.pjoints = 20
        self.ndimensions = 3


        self.load_preprocessed(self.data_dir)
        self.reset_batch_pointer()



    def load_preprocessed(self, data_dir):
        self.dance_data = []
        self.valid_dance_data = []
        for dance_file in os.listdir(self.data_dir):
            csvfile = open(self.data_dir + "/" + dance_file,"rt", encoding='utf-8')
            self.raw_dance_data = []
            dancereader = csv.reader(csvfile, delimiter=',')
            data_single_set = [] # one step contains ndimension*pjoint points
            data_single_tstep = [] # one set contains tsteps number of movements

            next(dancereader) #remove header
            for row in dancereader:
                if int(row[3]) == 0:
                    #every self.pjoints (20) is one timestamp of data
                    #if len(data_single_set) != 0:
                        #data_single_tstep.append(np.array(data_single_set))
                        #if len(data_single_tstep) % (self.tsteps+1) == 0 and len(data_single_tstep) != 0: #if we've filled our set of tstep movememnts
                        #    self.raw_dance_data.append(np.array(data_single_tstep))
                        #    data_single_tstep = []
                    if len(data_single_set) != 0:
                        single_set = np.array(data_single_set[:-1]).flatten().reshape(self.pjoints*self.ndimensions)
                        single_set = np.append(single_set, data_single_set[-1])
                        data_single_tstep.append(single_set)

                        if len(data_single_tstep) % (self.tsteps+1) == 0 and len(data_single_tstep) != 0: #if we've filled our set of tstep movememnts
                            self.raw_dance_data.append(np.array(data_single_tstep))
                            data_single_tstep = []

                    data_single_set = []
                #data_single_step.append([row[2], row[4], row[5], row[6], row[7]]) #relative time, position_index, x, y, z
                #data_single_set.append([row[3], row[4], row[5]])
                data_single_set.append([float(row[4]), float(row[5]), float(row[6])])
                if int(row[3]) == 19:
                    data_single_set.append(int(row[2]))

            csvfile.close()



            # every 1 in 20 (5%) will be used for validation data
            cur_data_counter = 0
            #print len(self.raw_dance_data)
            for i in range(len(self.raw_dance_data)):
                data = self.raw_dance_data[i]

                if cur_data_counter % 20 == 0:
                    self.valid_dance_data.append(data)
                else:
                    self.dance_data.append(data)
                cur_data_counter = cur_data_counter + 1

        #self.dance_data = np.array(self.dance_data)
        #self.valid_dance_data = np.array(self.valid_dance_data)


        # minus 1, since we want the ydata to be a shifted version of x data
        self.num_batches = int(len(self.dance_data) / self.batch_size)
        print("\tloaded dataset:")
        print("\t\t{} train individual data points".format(len(self.dance_data)))
        print("\t\t{} valid individual data points".format(len(self.valid_dance_data)))
        print("\t\t{} batches".format(self.num_batches))

    def get_train_length(self):
        return len(self.dance_data)

    def get_val_length(self):
        return len(self.valid_dance_data)

    def validation_data(self, batch=True):
        # returns validation data
        x_batch = []
        y_batch = []

        if batch==True: # return a batch
            for i in range(self.batch_size):
                valid_ix = i%len(self.valid_dance_data)
                data = self.valid_dance_data[valid_ix]
                x_batch.append(np.copy(data[:self.tsteps]))
                y_batch.append(np.copy(data[1:self.tsteps+1]))
        else: # return all
            for i in range(self.get_val_length()):
                valid_ix = i%len(self.valid_dance_data)
                data = self.valid_dance_data[valid_ix]
                x_batch.append(np.copy(data[:self.tsteps]))
                y_batch.append(np.copy(data[1:self.tsteps+1]))

        return x_batch, y_batch

    def next_batch(self, batch=True):
        # returns a randomized, tsteps-sized portion of the training data
        x_batch = []
        y_batch = []
        #ascii_list = []
        if batch == True:
            for i in range(self.batch_size):
                data = np.array(self.dance_data[self.idx_perm[self.pointer]])
                x_batch.append(np.copy(data[:self.tsteps]))
                y_batch.append(np.copy(data[1:self.tsteps+1]))
                self.tick_batch_pointer()
        else:
            for i in range(self.get_train_length()):
                data = np.array(self.dance_data[self.idx_perm[self.pointer]])
                x_batch.append(np.copy(data[:self.tsteps]))
                y_batch.append(np.copy(data[1:self.tsteps+1]))
                self.tick_batch_pointer()
        return x_batch, y_batch

    def tick_batch_pointer(self):
        self.pointer += 1
        if (self.pointer >= len(self.dance_data)):
            self.reset_batch_pointer()
    def reset_batch_pointer(self):
        self.idx_perm = np.random.permutation(len(self.dance_data))
        self.pointer = 0

    def get_data(self):
        #data = np.array(self.dance_data)
        #return np.copy(data[:][:][:self.tsteps])
        return self.dance_data

if __name__ == "__main__":
    dl = DataLoader()
    x_batch, y_batch = dl.next_batch(batch=False)
    xval, yval = dl.validation_data(batch=False)
    raw_data = dl.get_data()
    print(np.array(x_batch).shape)
    print(np.array(xval).shape)
    trans = np.transpose(raw_data,[0,2,1])
    print(trans)
    print(np.array(trans).shape)