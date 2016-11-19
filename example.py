import Image_Processing.imgpreprocess as igp
import Image_Processing.segpreprocess as sgp
from PIL import Image
import numpy as np
import os
k = igp.Preprocess("testimage.png")
lines = k.segment_lines()
if not os.path.exists("Seg_image"):
    os.makedirs("Seg_image")
for j in range(len(lines)):
    chars = igp.segment_characters(lines[j])
    
    for i in range(len(chars)):
        char = np.asarray(chars[i])*255
        char = Image.fromarray(char)
        char.save("Seg_image/"+str(j)+"_"+str(i)+".png")
