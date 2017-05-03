
# In[1]:

import tensorflow as tf
import numpy as np

# preprocessed data
#from datasets.twitter import data
import data_utils


def predicted_bpm(predicted_output_seq):
    winSize = 0.050
    maxBeatTime = int(round(2.0 / winSize))
    HistAll = np.zeros((maxBeatTime,))
    pos1 = predicted_output_seq
    posDifs = []                                                        # compute histograms of local maxima changes
    for j in range(len(pos1)-1):
        posDifs.append(pos1[j+1]-pos1[j])
    [HistTimes, HistEdges] = np.histogram(posDifs, np.arange(0.5, maxBeatTime + 1.5))
    HistCenters = (HistEdges[0:-1] + HistEdges[1::]) / 2.0
    HistTimes = HistTimes.astype(float) / max(pos1) #stFeatures.shape[1]
    HistAll += HistTimes



    # Get beat as the argmax of the agregated histogram:
    I = np.argmax(HistAll)
    BPMs = 60 / (HistCenters * winSize)
    BPM = BPMs[I]
    return BPM

def load_data(PATH=''):
    # read numpy arrays
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    return idx_q, idx_a
# load data from pickle and npy files
idx_q, idx_a = load_data(PATH='datasets/twitter/')
#idx_q = np.array([[2, 3, 2, 3, 2], [4, 3, 2, 3, 4], [4, 3, 3, 3, 4]])
#idx_a = np.array([[1, 0, 1, 0, 1], [1, 0, 1, 0, 1], [1, 1, 0, 0, 1]])
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters 
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 32
xvocab_size = 3840
yvocab_size = 2
emb_dim = 1024

import seq2seq_wrapper

# In[7]:

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/twitter/',
                               emb_dim=emb_dim,
                               num_layers=3
                               )


# In[8]:

val_batch_gen = data_utils.rand_batch_gen(validX, validY, 32)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)


# In[9]:
sess = model.restore_last_session()
sess = model.train(train_batch_gen, val_batch_gen)
