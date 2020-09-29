# -*- coding: utf-8 -*-
# @Time    : 2020/9/2 14:10
# @Author  : Haiyan Tan 
# @File    : gaussian.py

from scipy.ndimage import gaussian_filter
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side
ascent = misc.ascent()
result = gaussian_filter(ascent, sigma=3)
# ax1.imshow(ascent)
# ax2.imshow(result)
# plt.show()
# print(list(zip([1, 2, 3], [1, 2, 3])))
# print((1, 1) + (1,))
x1 = np.arange(9.0).reshape((3, 3))
print(x1)
x2 = np.arange(3.0)
print(x2)
r = np.subtract(x1, x2)
print(r)
print("28", np.ones((0, 5)))
