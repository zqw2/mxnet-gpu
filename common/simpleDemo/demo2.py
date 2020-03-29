from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random
from d2lzh import *
import numpy as np

# a = np.arange(0, 100, 10)
# print(a)
# b = np.where(a < 50, 1, 0)
# c = np.where(a >= 50)[0]
# print(b) # >>>(array([0, 1, 2, 3, 4]),)
# print(c) # >>>[5 6 7 8 9]


import numpy as np
import scipy
import scipy.misc
import scipy.ndimage

import imageio
import matplotlib.pyplot as plt

# img = imageio.imread('test.jpg')
# img_tinted = img * [1, 0.95, 0.9]
#
# # Show the original image
# plt.subplot(1, 2, 1)
# plt.imshow(img)
#
# # Show the tinted image
# plt.subplot(1, 2, 2)
#
# # A slight gotcha with imshow is that it might give strange results
# # if presented with data that is not uint8. To work around this, we
# # explicitly cast the image to uint8  before displaying it.
# plt.imshow(np.uint8(img_tinted))
# plt.show()


A = np.array([[2,1,-2],[3,0,1],[1,1,-1]])
b = np.transpose(np.array([[-3,5,-2]]))

x = np.linalg.solve(A,b)

print(x)


# x = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
#
# print(np.sum(x))  # Compute sum of all elements; prints "10"
# print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
# print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"
# print(np.sum(x, axis=2))  # Compute sum of each row; prints "[3 7]"
#

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# print(quicksort([3,6,8,10,1,2,1]))
# # Prints "[1, 1, 2, 3, 6, 8, 10]"