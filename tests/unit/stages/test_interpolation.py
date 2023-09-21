from __future__ import annotations

import numpy as np

from upp.stages.interpolation import (
    subdivide_bins,
    upscale_array,
    upscale_array_regionally,
)


def test_subdivide_bins():
    bins = np.array([0, 1, 2, 3, 4])
    n = 2
    expected_output = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    result = subdivide_bins(bins, n)
    np.testing.assert_array_equal(result, expected_output)


def test_upscale_array():
    # Create a sample input array
    input_array = np.array(
        [
            [
                1.0,
                2.0,
            ],
            [
                3.0,
                4.0,
            ],
        ]
    )
    expected_output = np.array(
        [
            [1.0, 1.25, 1.75, 2.0],
            [1.5, 1.75, 2.25, 2.5],
            [2.5, 2.75, 3.25, 3.5],
            [3.0, 3.25, 3.75, 4.0],
        ]
    )
    # Set parameters for upscale
    upscl = 2
    order = 1
    mode = "nearest"
    positive = True

    # Call the upscale_array function
    result = upscale_array(
        input_array,
        upscl,
        order=order,
        mode=mode,
        positive=positive,
    )

    # Check if the result matches the expected output
    np.testing.assert_array_equal(result, expected_output)


def test_upscale_array_regionally():
    # Create a sample input array
    input_array = np.array(
        [
            [
                1.0,
                2.0,
                7.0,
            ],
            [
                3.0,
                4.0,
                7.0,
            ],
            [
                8.0,
                8.0,
                9.0,
            ],
        ]
    )
    expected_output = np.array(
        [
            [
                1.0,
                1.25,
                1.75,
                2.0,
                7.0,
                7.0,
            ],
            [
                1.5,
                1.75,
                2.25,
                2.5,
                7.0,
                7.0,
            ],
            [
                2.5,
                2.75,
                3.25,
                3.5,
                7.0,
                7.0,
            ],
            [
                3.0,
                3.25,
                3.75,
                4.0,
                7.0,
                7.0,
            ],
            [
                8.0,
                8.0,
                8.0,
                8.0,
                9.0,
                9.0,
            ],
            [
                8.0,
                8.0,
                8.0,
                8.0,
                9.0,
                9.0,
            ],
        ]
    )
    # Set parameters for upscale
    upscl = 2
    order = 1
    mode = "nearest"
    positive = True

    # Call the upscale_array function
    result = upscale_array_regionally(
        input_array,
        upscl,
        order=order,
        mode=mode,
        positive=positive,
        regionlengthsd=[[2, 1], [2, 1]],
    )

    # Check if the result matches the expected output
    np.testing.assert_array_equal(result, expected_output)
