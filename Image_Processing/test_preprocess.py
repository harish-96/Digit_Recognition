import unittest
from Image_Processing.imgpreprocess import *
import numpy


class Test_Preprocess(unittest.TestCase):

    def setUp(self):
        self.imagepath = "testimage.jpg"
        self.PP = Preprocess(self.imagepath)

    def test_Preprocess_raises_error_for_invalid_file_or_not_found(self):
        """Check that Preprocess raises error for invalid file name 
        or file not found or a non image file"""
        self.assertRaises(IOError, Preprocess, "somefilename")
        self.assertRaises(IOError, Preprocess, "README.md")
        self.assertRaises(TypeError, Preprocess, 56)

    def test_binaryimg_outputs_a_binary_matrix(self):
        """Check that binaryimg function actually converts to binary and
        gives output as matrix of zeros and ones"""
        binaryimage = binaryimg(self.PP.image)
        self.assertEqual(type(binaryimage), numpy.matrixlib.defmatrix.matrix)
        self.assertTrue(numpy.all(binaryimage) in [0, 1])

    def test_segment_lines(self):
        """Check for a testimage that segment_lines is giving 
        correct number of lines as in image""" 
        lines = self.PP.segment_lines()
        self.assertEqual(len(lines), 5)

    def test_total_number_of_digits(self):
        """Check that segmentation is correct by checking if total number
        of digits in image is same as got by this module"""
        lines = self.PP.segment_lines()
        nchars = 0
        for line in lines:
            char = segment_characters(line)
            nchars = nchars + len(char)
        self.assertEqual(nchars, 50)

    def test_segment_characters_from_line(self):
        """Check that segmentation of characters from a line is correct by 
        checking total number of characters in a line of image is equal to that
        obtained by code"""
        lines = self.PP.segment_lines()
        char0 = segment_characters(lines[0])
        self.assertEqual(len(char0), 10)

    def tearDown(self):
        del self.PP


if __name__ == '__main__':
    unittest.main()
