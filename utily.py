# How is the CIFAR-10 and CIFAR-100 data collected == https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

from six.moves import cPickle as pickle
import numpy as np
import os
import platform
import matplotlib.pyplot as plt
#from scipy.misc import imread
# from scipy.misc.pilutil import imread
#https://stackoverflow.com/questions/9298665/cannot-import-scipy-misc-imread

# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict


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
    #print(datadict['batch_label']) # training batch 1 of 5 >> training batch 5 of 5
    #print(type(datadict['labels'])) # <class 'list'> - SIX Lists of Labels 
    #print(len(datadict['labels'])) # SIX Lists of Labels  - each of Length 10,000
    X = datadict['data'] # the numpy.ndarray with data --- ? 
    #print(type(X)) # <class 'numpy.ndarray'>
    Y = datadict['labels']

    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float") # FOO_BAR_TBD
    Y = np.array(Y)
    #print(Y.shape) #(10000,)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    # iterate through the - 5 - TRAINING BATCHES 
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte

# Load the raw CIFAR-10 data.
def loadCIFAR():
    cifar10_dir = '/home/dhankar/temp/cifar/CIFAR_kNN_Stanford/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    print('Training data - X_train -- shape: ', X_train.shape)
    print('Training labels - y_train -- shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape) 
    """
    Training data shape:  (50000, 32, 32, 3)
    Training labels shape:  (50000,)
    Test data shape:  (10000, 32, 32, 3)
    Test labels shape:  (10000,)
    """ 
    return X_train, y_train, X_test, y_test




