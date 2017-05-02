import numpy as np
import math
import random
import os
import cPickle as pickle
import xml.etree.ElementTree as ET
import csv 
import sklearn.preprocessing as preprocessing
import sys

from utils import *

class DataLoader():
    def __init__(self, args, logger, limit = 500):
        self.data_dir = args.data_dir
        self.alphabet = args.alphabet
        self.batch_size = args.batch_size
        self.tsteps = args.tsteps
        self.data_scale = args.data_scale # scale data down by this factor
        self.ascii_steps = args.tsteps/args.tsteps_per_ascii
        self.logger = logger
        self.limit = limit # removes large noisy gaps in the data
        self.pjoints = args.pjoints
        self.ndimensions = args.ndimensions
        self.data_mean = np.zeros(3)
        self.data_variance = np.zeros(3)
        self.sequence_shift = args.sequence_shift
        #self.load_preprocessed(self.data_dir)
        self.load_preprocessed(self.data_dir)
        self.reset_batch_pointer()



    def load_preprocessed(self, data_file):
        self.dance_data = []
        self.valid_dance_data = []
        self.raw_dance_data = []
        for dance_file in os.listdir(self.data_dir):
            csvfile = open(self.data_dir + "/" + dance_file,"rb")
            
            dancereader = csv.reader(csvfile, delimiter=',')
            data_single_set_x = [] # one step contains ndimension*pjoint points
            data_single_set_y = []
            data_single_tstep = [] # one set contains tsteps number of movements

            #grab overlapping sequences
            entire_file = []
            for row in dancereader:
                if len(row) == 10:
                    if row[2].isdigit(): # rule out the first line
                        if int(row[3]) == 0: 
                        
                            #every self.pjoints (20) is one timestamp of data 
                            if len(data_single_set_x) != 0:
                                #print np.concatenate((np.array(data_single_set_x), np.array(data_single_set_y)), axis=0).shape
                                entire_file.append(np.concatenate((np.array(data_single_set_x), np.array(data_single_set_y)), axis=0))
                                    
                            data_single_set_x= []
                            data_single_set_y = []
                        data_single_set_x.append(row[8])
                        data_single_set_y.append(row[9])
                                   
            csvfile.close()

            count = 0
            #each time, we take sequence from count*sequence_shift to count*sequence_shift + tsteps 
            entire_sequence_length = len(entire_file)
            while(True):
                upper_limit = count*self.sequence_shift+self.tsteps+1
                if (upper_limit>entire_sequence_length):
                    break
                lower_limit = count*self.sequence_shift
                self.raw_dance_data.append(np.array(entire_file[lower_limit:upper_limit]))
                count+=1
            print len(self.raw_dance_data)
            

            #grab nonoverlaping sequences
            '''
            for row in dancereader:
                if len(row) == 10:
                    if row[2].isdigit(): # rule out the first line
                        if int(row[3]) == 0: 
                        
                            #every self.pjoints (20) is one timestamp of data 
                            if len(data_single_set_x) != 0:
                                #print np.concatenate((np.array(data_single_set_x), np.array(data_single_set_y)), axis=0).shape
                                data_single_tstep.append(np.concatenate((np.array(data_single_set_x), np.array(data_single_set_y)), axis=0))

                                if len(data_single_tstep) % (self.tsteps+1) == 0 and len(data_single_tstep) != 0: #if we've filled our set of tstep movememnts
                                    self.raw_dance_data.append(np.array(data_single_tstep))
                                    
                                    data_single_tstep = []
                            data_single_set_x= []
                            data_single_set_y = []
                        #data_single_step.append([row[2], row[4], row[5], row[6], row[7]]) #relative time, position_index, x, y, z
                        #data_single_set.append([row[3], row[4], row[5]])
                        data_single_set_x.append(row[8])
                        data_single_set_y.append(row[9])
                       
                        
            csvfile.close()
            '''

        # every 1 in 20 (5%) will be used for validation data
        cur_data_counter = 0
        #print len(self.raw_dance_data)
        for i in range(len(self.raw_dance_data)):
            data = self.raw_dance_data[i]

            
            if cur_data_counter % 20 == 0:
                self.valid_dance_data.append(data)
            else:
                self.dance_data.append(data)
            
            #self.dance_data.append(data)
            #self.valid_dance_data.append(data)
            cur_data_counter = cur_data_counter + 1

        self.dance_data = np.array(self.dance_data, dtype=np.float32)
        self.valid_dance_data = np.array(self.valid_dance_data, dtype=np.float32)
        print self.valid_dance_data.shape
        print self.dance_data.shape

        '''
        csvfile = open('data.csv', 'a+')
        csvwriter = csv.writer(csvfile, delimiter=",")
        csvwriter.writerow(["x", "y"])
        for  row in self.valid_dance_data[0]:
            for i in range(20):

                csvwriter.writerow([row[i*2], row[i*2+1]])

        sys.exit()
        '''



        # minus 1, since we want the ydata to be a shifted version of x data
        self.num_batches = int(len(self.dance_data) / self.batch_size)
        print("\tloaded dataset:")
        print("\t\t{} train individual data points".format(len(self.dance_data)))
        print("\t\t{} valid individual data points".format(len(self.valid_dance_data)))
        print("\t\t{} batches".format(self.num_batches))


    def validation_data(self):
        # returns validation data
        x_batch = []
        y_batch = []
        
        for i in range(self.batch_size):
            valid_ix = i%len(self.valid_dance_data)
            data = self.valid_dance_data[valid_ix]
            #print data.shape
            #print data[1:self.tsteps+1].shape
            x_batch.append(np.copy(data[:self.tsteps]))
            y_batch.append(np.copy(data[1:self.tsteps+1]))
        '''
        csvfile = open('data.csv', 'a+')
        csvwriter = csv.writer(csvfile, delimiter=",")
        csvwriter.writerow(["x", "y"])
        for  row in self.valid_dance_data[0]:
            for i in range(20):

                csvwriter.writerow([row[i*2], row[i*2+1]])

        sys.exit()
        '''
        return x_batch, y_batch

    def next_batch(self):
        # returns a randomized, tsteps-sized portion of the training data
        x_batch = []
        y_batch = []
        #ascii_list = []
        for i in xrange(self.batch_size):
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


# abstraction for logging
class Logger():
    def __init__(self, args):
        self.logf = '{}train_scribe.txt'.format(args.log_dir) if args.train else '{}sample_scribe.txt'.format(args.log_dir)
        #with open(self.logf, 'w') as f: #f.write("Scribe: Realistic Handriting in Tensorflow\n     by Sam Greydanus\n\n\n")

    def write(self, s, print_it=True):
        if print_it:
            print s
        with open(self.logf, 'a') as f:
            f.write(s + "\n")
