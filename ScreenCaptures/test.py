## Source -- https://numpy.org/doc/stable/reference/generated/numpy.reshape.html

import numpy as np 

arr = np.zeros((5, 2)) # 5 Rows and 2 Cols of ZERO's 
print(arr)
b_arr = arr.T
print(b_arr)
v_b_arr = b_arr.view() # a VIEW of the Array - not the actual b_arr
#v_b_arr.shape = (20) # ValueError: cannot reshape array of size 10 into shape (20,)
#v_b_arr.shape = (10) # AttributeError: incompatible shape for a non-contiguous array
#res_arr = np.reshape(arr, (2, 3)) # C-like index ordering
#print(res_arr)
#
#
arr1 = np.arange(6).reshape((3, 2))
print(arr1)
arr1.reshape((2,3))
print(arr1)
#
# Slicing 2D Array -- 3 Rows and 3 Cols
arr2 = np.arange(9).reshape((3, 3))
print(arr2)
print("-- "*20)
print(arr2[:-1,:-1]) # DROP - Last ROW , also DROP Last COL 
print("-- "*20)
print(arr2[:1,:-1]) # PRINT only - First ROW , also DROP Last COL 
print("-- "*20)
print(arr2[:1,:1]) # PRINT only - First ROW and First COL 
print("-- "*20)
print(arr2[:3,2:3]) # PRINT ALL 3 ROWS , but Only LAST COL 
print("-- "*20)
print(arr2[:3,-1]) # PRINT ALL 3 ROWS , but Only LAST COL == [2 5 8]
#...this is a 1D Array a vector of the Last COlumn Values 
#
# Reorganizing Arrays 
#
# 2D Array -- 3 Rows and 3 Cols
print("-- "*20)
arr3 = np.arange(5,15) # [ 5  6  7  8  9 10 11 12 13 14] # 15 EXCLUDED 
arr3 = np.arange(5,14).reshape((3, 3))
print(arr3)
#
print("-- "*20)

















