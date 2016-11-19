import random
from Neural_Network import text_recog
from center_image import *
# Third-party libraries
import numpy as np
from numpy import array, argwhere
from PIL import Image
from PIL import ImageOps
from scipy import ndimage
from skimage.morphology import erosion, dilation
def feedImage(image):
    '''takes an image and attempts to predict the number present using
        a neural network and pre-processing methods'''
    image =  gradual_normalization(image)
    (x, y) = ndimage.measurements.center_of_mass(np.array(image))
    large_window = Image.new("L", (28,28))
    large_window.paste(image, (int(14-x),int(14-y)))
    image_matrix =  standardize_image(large_window, False)/255
    image_matrix = center_image(image_matrix)
    input_vector = np.reshape(image_matrix, (784, 1))
    return input_vector


def gradual_normalization(  image):
    #adjusts sizing and stroke to maintain consistency
    while(image.size[0] > 50 and image.size[1] > 50):
        image = image.resize((image.size[0]/2, image.size[1]/2), Image.ANTIALIAS)
        image =  stroke_normalization(image)
    #image_matrix = image/255.0
    image_matrix =  standardize_image(image)/255
    
    #stores indices of non-zero numbers
    non_zeros = argwhere(image_matrix)
    (ystart, xstart), (ystop, xstop) = non_zeros.min(0), non_zeros.max(0) + 1
    #prevents stretching the number one to a square 
    if(xstop - xstart > 8):
        image_matrix = image_matrix[ystart:ystop, 0:image.size[0]]
    image = Image.fromarray(np.int8(image_matrix*255)).convert('L')
    width = 20
    height = 20
    image = image.resize((width, height), Image.ANTIALIAS)
    return image

def stroke_normalization(  image):        
    '''takes an image and checks if thickness is within a
        threshold, then uses erosion/dilation to adjust thickness
        appropriately'''   
    #image_matrix = image/255    
    image_matrix =  standardize_image(image)/255
    thickness =  stroke_thickness(image_matrix)
    while(abs(thickness-3.3806208986) > 1.0):
        #erodes/dialates binary image, ten calculates the binary image thickness
        #converts np.bool type to int8 (data able to be handled by PIL)
        if(thickness > 3.3806208986):
            image_matrix = ndimage.morphology.binary_erosion(image_matrix)
            thickness =  stroke_thickness(Image.fromarray(np.int8(image_matrix)))
        else:
            image_matrix = ndimage.morphology.binary_dilation(image_matrix)
            thickness =  stroke_thickness(Image.fromarray(np.int8(image_matrix)))
    image_matrix = ndimage.morphology.binary_closing(image_matrix)
    transformed_image = ImageOps.invert(Image.fromarray(np.int8(image_matrix*255)).convert('L'))
    return transformed_image
    
def stroke_thickness(  image_matrix):
    '''calculates the thickness of stroke in an image matrix'''
    #euclidean distance transformation
    image_edt = ndimage.distance_transform_edt(image_matrix)
    image_edt *= 10

    dist_values = [i for i in image_edt]
    thickness = 2*np.mean(dist_values)
    return thickness

def standardize_image(  image, invert=True):
    '''resizes image and turns it black and white'''
    image = image.convert('L')
    if(invert):
        image = ImageOps.invert(image)
    #convert image to black and white
    image = image.point(lambda x: 0 if x<64 else 255, 'L')
    return np.array(image)
