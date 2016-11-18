import Image_Processing.imgpreprocess as igp
import cv2
import numpy as np
k = igp.Preprocess("testimage.jpg")
lines = k.segment_lines()
charsinl0 = igp.segment_characters(lines[0])
for i in charsinl0:
    char028 = np.zeros((28,28))
    char0 = cv2.resize(i,(20,20))
    for i in range(20):
        for j in range(20):
            char028[4+i][4+j] = char0[i][j]
    char028re = np.reshape(char028,(784,1))
    ex = nn.forward_prop(char028re)[0][-1]
    print np.argmax(ex)
    