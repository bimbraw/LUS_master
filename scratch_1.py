import sys
import cv2

import numpy as np
import skimage.color
import skimage.io
from matplotlib import pyplot as plt

# read image, based on command line filename argument;
# read the image as grayscale from the outset
img = cv2.imread("selfie.jpg")

# display the image
#cv2.imshow('image', image)
#cv2.waitKey(0)

# create the histogram
histogram, bin_edges = np.histogram(img, bins=256, range=(0, 1))

print(histogram)


# configure and draw the histogram figure
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim([0.0, 1.0])  # <- named arguments do not work here
plt.axvline(img.mean(), color='k', linestyle='dashed', linewidth=1)

plt.plot(bin_edges[0:-1], histogram)  # <- or here
plt.show()

value = (img.mean())*255
print(value)

ret,thresh1 = cv2.threshold(img,32,255,cv2.THRESH_BINARY)
cv2.imshow('gray', thresh1)
cv2.waitKey(0)

