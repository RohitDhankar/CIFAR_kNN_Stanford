import numpy as np
import os
import matplotlib.pyplot as plt

# Dummy data to create a CIFAR shaped array 
arr = np.arange(880).reshape((40,66))
#print(arr)
arr1 = arr.reshape(40, 3, 32, 32).transpose(0,2,3,1).astype("float") 
print(arr1)