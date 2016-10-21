import unittest
from imgpreprocess import *
import numpy

class Test_Preprocess(unittest.TestCase):
    def setUp(self):
        self.imagepath = "testimage2.jpg"
        self.PP = Preprocess(self.imagepath)
    def test_Preprocess_raises_error_for_invalid_file_or_not_found(self):
        self.assertRaises(IOError , Preprocess , "somefilename")
        self.assertRaises(IOError , Preprocess , "README.md")
        self.assertRaises(TypeError , Preprocess , 56)
    def test_binaryimg_outputs_a_binary_matrix(self):
        binaryimage = binaryimg(self.PP.image)
        self.assertEqual(type(binaryimage),numpy.matrixlib.defmatrix.matrix)
        self.assertTrue(numpy.all(binaryimage) in [0,1])
    def test_cropimg_outputs_matrix(self):
        self.assertEqual(type(cropimg(binaryimg(self.PP.image))),numpy.matrixlib.defmatrix.matrix)
    def test_segment_lines_outputs_list_of_matrices(self):
        lines = self.PP.segment_lines()
        self.assertEqual(type(lines),list)
        self.assertEqual(type(lines[0]),numpy.matrixlib.defmatrix.matrix)
        
    def tearDown(self):
        del self.PP


if __name__ == '__main__':
    unittest.main()            
            
 