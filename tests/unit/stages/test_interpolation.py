import unittest

import numpy as np
from scipy import ndimage

from upp.stages.interpolation import (
    subdivide_bins,
    upscale_array,
    upscale_array_regionally,
)


class TestSubdivideBins(unittest.TestCase):

    def test_subdivide_bins(self):
        bins = np.array([0, 1, 2, 3, 4])
        n = 2
        expected_output = np.array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4.])
        
        result = subdivide_bins(bins, n)
        
        np.testing.assert_array_equal(result, expected_output)

class TestUpscaleArray(unittest.TestCase):

    def test_upscale_array(self):
        # Create a sample input array
        input_array = np.array([[1., 2.,],[3., 4.,]])
        expected_output = np.array([[1.  , 1.25, 1.75, 2.  ],
       								[1.5 , 1.75, 2.25, 2.5 ],
									[2.5 , 2.75, 3.25, 3.5 ],
									[3.  , 3.25, 3.75, 4.  ]])
        # Set parameters for upscale
        upscl = 2
        order = 1
        mode = "nearest"
        normalise = False
        positive = True

        # Call the upscale_array function
        result = upscale_array(input_array, upscl, order=order, mode=mode, normalise=normalise, positive=positive)

        # Check if the result matches the expected output
        np.testing.assert_array_equal(result, expected_output)

class TestUpscaleArrayRegionally(unittest.TestCase):
    def test_upscale_array(self):
        # Create a sample input array
        input_array = np.array([[1., 2., 7.,], [3., 4., 7.,], [8., 8., 9.,]])
        expected_output = np.array([[1.  , 1.25, 1.75, 2.,  7., 7., ],
       								[1.5 , 1.75, 2.25, 2.5 , 7., 7., ],
									[2.5 , 2.75, 3.25, 3.5 , 7., 7., ],
									[3.  , 3.25, 3.75, 4. , 7., 7., ],  
                                	[8., 8., 8., 8., 9., 9., ],
									[8., 8., 8., 8., 9., 9., ]])									
        # Set parameters for upscale
        upscl = 2
        order = 1
        mode = "nearest"
        normalise = False
        positive = True

        # Call the upscale_array function
        result = upscale_array_regionally(input_array, upscl, order=order, mode=mode, normalise=normalise, positive=positive, regionlengthsd=[[2, 1], [2, 1]])

        # Check if the result matches the expected output
        np.testing.assert_array_equal(result, expected_output)

if __name__ == '__main__':
    unittest.main()

