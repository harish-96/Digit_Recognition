"""
imgpreprocess.py
~~~~~~~~~~~~~~~~
A module to preprocess and segement an image containing text 
into lines , words and characters.And also change resolution of 
the segmented characters to the resolution required by the NeuralNet.py 
to recognise the characters.
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
class Preprocess(object):
    def __init__(self,imagepath):
	"""imagepath is the address of the image that needs to be preprocessed"""
        self.image = cv2.imread(str(imagepath),0)
    def binaryimg(self,image):
	"""Converts grayscale image to binary image , it takes 1 for black and it takes zero for white """
        blur_image = cv2.GaussianBlur(image,(5,5),0)
        retval,binary_image = cv2.threshold(blur_image,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        m, n = binary_image.shape
        binary_image = np.asmatrix(binary_image)
        for i in range(n):
            for j in range(m):
                if binary_image[j,i]==0:
                    binary_image[j,i]=1
        for i in range(n):
            for j in range(m):
                if binary_image[j,i]==255:
                    binary_image[j,i]=0
        return binary_image
    def cropimg(self,image):
	"""Crops a binary image tightly"""
        m,n = image.shape
        c1 = n 
        c2 = 0
        r1 = m
        r2 = 0
        for i in range(n):
            for j in range(m):
                if image[j,i]==1: 
                    if i < c1:
                        c1 =i
                    if i > c2:
                        c2 = i
                    if  j <r1 :
                        r1 = j
                    if  j > r2 :
                        r2 = j
        return image[r1:r2+1,c1:c2+1]
    def segment_lines(self):
	"""The image containing text is segmented into lines and returns a list of the lines""" 
        cropped_image = self.cropimg(self.binaryimg(self.image))
        lines = []
        limg = cropped_image.copy()
        last_line = self.last_line(limg)
        while not np.sum(limg) == np.sum(last_line):
            m,n = limg.shape
            p = 0
            for i in range(1,m):
                if np.sum(limg[i,:])==0 and np.sum(limg[i-1,:])!=0:
                    lines.append(limg[:i,:])
                    p = i
                    break
            limg = limg[p:,:]
            limg = self.cropimg(limg)
        lines.append(last_line)
        return lines
    def last_line(self,img):
	"""Takes cropped binary image as input and gives the segment of image containing last line of text""" 
        m,n = img.shape
        for i in list(reversed(range(m-1))):
            if np.sum(img[i,:])==0 and np.sum(img[i+1,:])!=0:                
                last_line = img[i+1:,:]
                break
        return last_line
    def segment_characters(self,line):
	"""Takes a line from a segemented image and returns a list of characters in the line"""
        line = self.cropimg(line)
        chars = []
        cimg = line.copy()
        last_char = self.last_char(cimg)
        while not np.sum(cimg) == np.sum(last_char):
            m,n = cimg.shape
            p = 0
            for i in range(1,n):
                if np.sum(cimg[:,i])==0 and np.sum(cimg[:,i-1])!=0:
                    chars.append(cimg[:,:i])
                    p = i
                    break
            cimg = cimg[:,p:]
            cimg = self.cropimg(cimg)
        chars.append(last_char)
        return chars
    def last_char(self,img):
	"""Takes a line from a segemented image and returns last character in the line"""
        m,n = img.shape
        for i in list(reversed(range(n-1))):
            if np.sum(img[:,i])==0 and np.sum(img[:,i+1])!=0:                
                last_char = img[:,i+1:]
                break
        return last_char


