import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('testimage.jpg' , 0) # zero to load the image in grayscale
#ret,thresh1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #cv2.threshold(img,45,255,cv2.THRESH_BINARY)
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(th3,'gray')
plt.show()
