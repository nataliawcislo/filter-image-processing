import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# img = cv.imread('frame60.tiff', 0)
# # Initiate ORB detector
# orb = cv.ORB_create()
# # find the keypoints with ORB
# kp = orb.detect(img,None)
# # compute the descriptors with ORB
# kp, des = orb.compute(img, kp)
# # draw only keypoints location,not size and orientation
# img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
# plt.imshow(img2), plt.show()


#reading image
img1 = cv.imread('frame60.tiff')
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

#keypoints
sift = cv.xfeatures2d.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)

img_1 = cv.drawKeypoints(gray1,keypoints_1,img1)
plt.imshow(img_1)
plt.show()
