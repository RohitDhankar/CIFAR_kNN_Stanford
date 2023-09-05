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
    X = datadict['data'] # the numpy.ndarray with data --- ? 
    #print(type(X)) # <class 'numpy.ndarray'>
    #print(X.shape) #(10000, 3072)
    Y = datadict['labels']
    #print(type(Y)) # <class 'list'>
    print("--    "*10)
    for k in range(1):
        im_r = X[k,:1024].reshape(32, 32)/255.0
        #print(im_r.shape) #(32,32)
        #plt.imshow(im_r) # Image RED Channel 
        #plt.show()
        im_g = X[k,1024:2048].reshape(32, 32)/255.0
        im_b = X[k,2048:].reshape(32, 32)/255.0
        img_rgb = np.dstack((im_r, im_g, im_b))
        img_label = str(datadict['labels'][k]) # 6 == FROG , 9 == TRUCK
        print(img_label) # 6 , 9 , 9 
        plt.figure(figsize=(2,2))
        plt.xlabel(str(img_label)) # ?? FOOBAR_Not OK 
        plt.imshow(img_rgb,interpolation='nearest',aspect='auto') 
        plt.show()
        #

    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float") 
    # 3072 = 3*32*32
    Y = np.array(Y)
    #print(Y.shape) #(10000,)
    return X, Y


cifar10_dir = '/home/dhankar/temp/cifar/CIFAR_kNN_Stanford/cifar-10-batches-py'
X, Y = load_CIFAR_batch(os.path.join(cifar10_dir,'data_batch_1'))
#print(type(X)) # <class 'numpy.ndarray'>
#print(X.shape) # (10000, 32, 32, 3)


# Load the RAW Cifar Data
#X_train, y_train, X_test, y_test = loadCIFAR()

