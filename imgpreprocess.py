import cv2
import numpy as np
import matplotlib.pyplot as plt
class Segmentation(object):
    def __init__(self,imagepath):
        self.image = cv2.imread(str(imagepath),0)
    def binaryimg(self,image):
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
            limg = self.trim_top(limg)
        lines.append(last_line)
        return lines
    def last_line(self,img):
        m,n = img.shape
        for i in list(reversed(range(m-1))):
            if np.sum(img[i,:])==0 and np.sum(img[i+1,:])!=0:                
                last_line = img[i+1:,:]
                break
        return last_line
    def trim_top(self,img):
        m,n = img.shape
        r = m
        for i in range(n):
            for j in range(m):
                if img[j,i]==1 and j < r:
                    r =j
        return img[r:,:]
    def segment_characters(self,line):
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
            cimg = self.trim_left(cimg)
        chars.append(last_char)
        return chars
    def last_char(self,img):
        m,n = img.shape
        for i in list(reversed(range(n-1))):
            if np.sum(img[:,i])==0 and np.sum(img[:,i+1])!=0:                
                last_char = img[:,i+1:]
                break
        return last_char
    def trim_left(self,img):
        m,n = img.shape
        c = n
        for i in range(n):
            for j in range(m):
                if img[j,i]==1 and i < c:
                    c = i
        return img[c:,:]

