This project is about developing a program to to recognize text from digital images of handwritten text.

The rough process is as follows:

*The image is converted to binary using opencv

*Lines are recognized by considering only rows with atleast one non-zero element (1 as data is binary) 

Lets call rows with all elements as zeros as zero-rows and similarly columns , zero-columns.

*Groups of rows separated by zero-rows are separate lines

*In each line words are separated by zero-columns

*Characters in a word are also separated by zero-columns  but words are separated by more number of columns than that of characters

*The segmentation part is over and now the size of each segment containing a character is adjusted to the size of images of training data.

*Now each segment is fed to the neural network for finding  each character.From characters, words can be formed 

and from words, lines can be formed and with lines we can reproduce entire text from the image.  

