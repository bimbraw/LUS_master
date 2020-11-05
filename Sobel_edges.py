import cv2

img = cv2.imread("simple.jpg")
cv2.imshow('img',img)
cv2.waitKey(0)

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F, 0, 1, ksize=5)

print('Sobel X values - ')
print(sobelx)
print('Sobel Y values - ')
print(sobely)

cv2.imshow('sobelx',sobelx)
cv2.waitKey(0)

cv2.imshow('sobely',sobely)
cv2.waitKey(0)

cv2.destroyAllWindows()