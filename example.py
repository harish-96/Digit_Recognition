import Image_Processing.imgpreprocess as igp
import Image_Processing.segpreprocess as sgp
from Neural_Network import text_recog as tr
from PIL import Image
from PIL import ImageOps
import numpy as np
import os
weights = np.array([np.loadtxt("weights0.csv",delimiter=","),np.loadtxt("weights1.csv",delimiter=",")])
biases = np.array([np.loadtxt("biases0.csv",delimiter=","),np.loadtxt("biases1.csv",delimiter=",")])
biases[0]=np.reshape(biases[0],(biases[0].shape[0],1))
biases[1]=np.reshape(biases[1],(biases[1].shape[0],1))
def forward_prop(weights,biases,num_layers,x_train):
    """Computes the activations and weighted inputs of the neurons in
    the network for the given input data.

    :param ndarray x_train: The input for the first layer which needs to be forwards propogated
    :return: A tuple of lists containing activations and weighted inputs
    """

    activations = []
    z = []
    activations.append(x_train)

    for i in range(num_layers - 1):
        z.append(np.dot(weights[i], activations[-1]) + biases[i])
        activations.append(tr.sigmoid(z[-1]))

    return activations[1:], z

k = igp.Preprocess("testimage.jpg")
lines = k.segment_lines()
if not os.path.exists("Seg_image"):
    os.makedirs("Seg_image")
for j in range(len(lines)):
    chars = igp.segment_characters(lines[j])
    print "line no: "+str(j+1)
    for i in range(len(chars)):
        char = np.asarray(chars[i])*255
        char = Image.fromarray(char)
        imgpath = "Seg_image/"+str(j)+"_"+str(i)+".png"
        char.save(imgpath)
        image = Image.open(imgpath)
        image = ImageOps.invert(image)
        processed_img = sgp.feedImage(image)
        ex = forward_prop(weights,biases,3,processed_img)[0][-1]
        print np.argmax(ex)
        
