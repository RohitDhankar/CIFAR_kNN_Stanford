# Loading only 1 CIFAR Batch - experimenting with the load process 
from six.moves import cPickle as pickle
import numpy as np
import os
import platform
import matplotlib.pyplot as plt

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = load_pickle(f) # load_pickle = 
    #print(type(datadict)) # dict 
    #print(datadict.keys()) # dict_keys(['batch_label', 'labels', 'data', 'filenames'])
    #print(datadict['batch_label']) # training batch 1 of 5
    #print(type(datadict['labels'])) # <class 'list'> 
    #print(len(datadict['labels'])) # 10,000
    X = datadict['data'] # the numpy.ndarray with data --- ? 
    #print(type(X)) # <class 'numpy.ndarray'>
    #print(X.shape) # (10000, 3072)
    print(X[:3]) # Head of - numpy.ndarray
    print(X[-3:]) # Tail of - numpy.ndarray
    print(X[[0, 1, 2], :]) 
    # print - First 3 ROWS as these 3 ROWS have the R,G,B values for the First Image
    # This is same as above - print(X[:3]) # Head of - numpy.ndarray
    print(X[np.ix_([0,1,2], [0,1])])
    # Above np.ix_ - first 3 ROWS of first 2 COLUMNS
    Y = datadict['labels']
    #print(type(Y)) # <class 'list'>


    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float") # FOO_BAR_TBD
    Y = np.array(Y)
    #print(Y.shape) #(10000,)
    return X, Y


cifar10_dir = '/home/dhankar/temp/cifar/CIFAR_kNN_Stanford/cifar-10-batches-py'
X, Y = load_CIFAR_batch(os.path.join(cifar10_dir,'data_batch_1'))
#print(type(X)) # <class 'numpy.ndarray'>
#print(X.shape) # (10000, 32, 32, 3)


# Load the RAW Cifar Data
#X_train, y_train, X_test, y_test = loadCIFAR()

