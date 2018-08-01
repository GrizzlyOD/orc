from PIL import Image
import numpy as np
import os
import matplotlib.pylab as plt
import cv2

dir = "I:\\train_1000\\vert\\1_29.jpg"
img = Image.open(dir).convert('RGB')
img1 = np.array(img)
h = img1.shape[0]
w = img1.shape[1]
print(img1.shape)
center = (w/2,h/2)
angle = 90

M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(img1, M, (w,h))

print(type(rotated))

rotated.reshape(w,h,3)
print(rotated.shape)

plt.imshow(rotated)
plt.show()  # 看图

