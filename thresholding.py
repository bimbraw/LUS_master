import cv2
import numpy as np

from matplotlib import pyplot as plt

#method 1 - with median as threshold
img = cv2.imread("selfie.jpg", 0)
median = np.median(img)
print(median)

ret,thresh1 = cv2.threshold(img,median,255,cv2.THRESH_BINARY)

n_white_pix = np.sum(thresh1 == 255)
print('Number of white pixels for median:', n_white_pix)
n_black_pix = np.sum(thresh1 == 0)
print('Number of black pixels for median:', n_black_pix)
per_black = (n_black_pix/(n_black_pix + n_white_pix)) * 100
per_white = (n_white_pix/(n_black_pix + n_white_pix)) * 100
print('Percentage of black pixels is: ' + str(per_black) +
      '% and percentage of white pixels is: ' + str(per_white) + '% for median.')


#method 2 - with mean as threshold
mean = np.mean(img)
print(mean)

ret,thresh2 = cv2.threshold(img,mean,255,cv2.THRESH_BINARY)

n_white_pix = np.sum(thresh2 == 255)
print('Number of white pixels for mean:', n_white_pix)
n_black_pix = np.sum(thresh2 == 0)
print('Number of black pixels for mean:', n_black_pix)
per_black = (n_black_pix/(n_black_pix + n_white_pix)) * 100
per_white = (n_white_pix/(n_black_pix + n_white_pix)) * 100
print('Percentage of black pixels is: ' + str(per_black) +
      '% and percentage of white pixels is: ' + str(per_white) + '% for mean.')


#plotting both the results side by side
f = plt.figure()
f.add_subplot(2,1,1)
plt.imshow(thresh1, cmap="gray")
plt.title('With median threshold')
f.add_subplot(2,1,2)
plt.imshow(thresh2, cmap="gray")
plt.title('With mean threshold')
plt.show(block=True)

#Median gives a better output (values closer to 50% each)