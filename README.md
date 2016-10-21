This project is about developing a program to to recognize text from digital images of handwritten text.

The module imgpreprocess.py is used for preprocessing image. 

The module text_recog.py is used for identifying characters from the preprocessed inages of characters

The rough process is as follows:

1)The image is converted to binary using opencv

2)Lines are recognized by considering only rows with atleast one non-zero element (1 as data is binary) 

Lets call rows with all elements as zeros as zero-rows and similarly columns , zero-columns.

3)Groups of rows separated by zero-rows are separate lines

4)In each line characters are separated by zero-columns

5)The segmentation part is over and now the size of each segment containing a character is adjusted to the size of images of training data.

6)Now each segment is fed to the neural network for finding  each character. 

  

