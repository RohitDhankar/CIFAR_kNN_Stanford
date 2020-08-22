# Are We There YET = http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html

import random
import numpy as np
import matplotlib.pyplot as plt

from utily import loadCIFAR

# Load the RAW Cifar Data
X_train, y_train, X_test, y_test = loadCIFAR()

def plotCifar():
    print(type(X_train))
    """
    # Visualize some examples from the dataset.
    # We show a few examples of training images from each class.
    """
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        #print(type(y)) #<class 'int'>
        idxs = np.flatnonzero(y_train == y) ##FOO_BAR_TBD--
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

"""
# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]
"""

