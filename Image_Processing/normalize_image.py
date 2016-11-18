import numpy as np


def Normalize(image):
	output_image = (image - np.mean(image))/np.std(image)
	return output_image