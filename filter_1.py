import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('f9.jpg')
#image = cv2.imread('1.jpeg')
# convert to black and white
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
r ,image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
# create kernel
kernel = np.ones((5,5), np.uint8)
fig, ax = plt.subplots(1, figsize=(16,12))
# original
ax = plt.subplot(232)
plt.imshow(image, cmap='Greys')
plt.title('original')
# erosion
e = cv2.erode(image, kernel)
ax = plt.subplot(234)
plt.imshow(e, cmap='Greys')
plt.title('erosion')
# dilation
d = cv2.dilate(image, kernel)
ax = plt.subplot(235)
plt.imshow(d, cmap='Greys')
plt.title('dilation')
# morphological gradient (dilation - erosion)
m = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
ax = plt.subplot(236)
plt.imshow(m, cmap='Greys')
plt.title('dilation - erosion')
plt.show()