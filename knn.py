import random
import numpy as np
import matplotlib.pyplot as plt

from utily import loadCIFAR

# Load the RAW Cifar Data
X_train, y_train, X_test, y_test = loadCIFAR()
"""
Training data - X_train -- shape:  (50000, 32, 32, 3)
Training labels - y_train -- shape:  (50000,)
Test data shape:  (10000, 32, 32, 3)
Test labels shape:  (10000,)
"""

def plotCifar():
    """
    # Visualize some examples from the dataset.
    # We show a few examples of training images from each class.
    """
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        #print(type(y)) #<class 'int'>
        #print(y) # 0 to 9 - 10 Classes 
        idxs = np.flatnonzero(y_train == y) ##FOO_BAR_TBD--
        #print(type(idxs)) # <class 'numpy.ndarray'> 
        #Output array, containing the indices of the elements of a.ravel() that are non-zero.
        #print(idxs) #[   29    30    35 ... 49941 49992 49994]
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()

plotCifar()    


# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask_train = list(range(num_training))
#print(len(mask_train)) #5000
X_train = X_train[mask_train]
#print(type(X_train)) #<class 'numpy.ndarray'>
#print(X_train.shape) # (5000, 32, 32, 3)
y_train = y_train[mask_train]

num_test = 500
mask_test = list(range(num_test))
X_test = X_test[mask_test]
y_test = y_test[mask_test]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape) # (5000, 3072) (500, 3072)

from knnCore import *

# Create a kNN classifier instance. 
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

#
# Test your implementation:
dists = classifier.compute_distances_two_loops(X_test)
# After about 10Secs delay and the terminal frozen - (500, 5000)
print(dists.shape) 
#
# We can visualize the distance matrix: each row is a single test example and
# its distances to training examples
plt.imshow(dists, interpolation='none')
plt.show()




