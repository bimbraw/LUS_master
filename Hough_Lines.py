import sys
import math
import cv2 as cv2
import numpy as np


def main(argv):
    default_file = 'C:/Users/Keshav Bimbraw/.PyCharmCE2019.3/config/scratches/katha.png'
    img = cv2.imread("katha.png")
    cv2.imshow('img', img)
    cv2.waitKey(0)

    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv2.imread(cv2.samples.findFile(filename), 0)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1

    blur = cv2.GaussianBlur(src, (5, 5), 0)

    #dst = cv2.Canny(blur, 50, 200, None, 3)
    dst = cv2.Laplacian(blur, cv2.CV_8UC1)
    #print(dst)
    #dst = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=5)

    # Copy edges to the images that will display the results in BGR
    dst.astype(np.float32)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLines(dst, 1, np.pi / 180, 350, None, 0, 0)#439 for 121,121, 431 for 21, 21, 435 for 11,11

    #print(lines[0,0,0], lines[0,0,1])

    dist = np.zeros(len(lines)+1)

    for i in range(0, len(lines)+1):
        dist[i] = np.linalg.norm(lines[i-1,0,0] - lines[i-1,0,1])

    print(type(dist))
    print(dist)

    dist.sort()
    print(dist)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("Source", src)
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)

    cv2.waitKey()
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])