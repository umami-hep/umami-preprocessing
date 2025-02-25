from __future__ import annotations

import numpy as np
from scipy import ndimage


def subdivide_bins(bins: np.array, n: int = 2) -> np.array:
    """Subdivide bins into n subbins.

    Parameters
    ----------
    bins : np.array
        array of bins edges to be subdivided
    n : int, optional
        Number of subbins, by default 2

    Returns
    -------
    np.array
        array of bin edges for the subdivided bins
    """
    comb_list = [np.array((bins[0],))]
    for i in range(len(bins) - 1):
        comb_list.append(np.linspace(bins[i], bins[i + 1], n + 1)[1:])
    return np.concatenate(comb_list)


def upscale_array(
    array: np.array,
    upscaling_factor: int,
    order: int = 3,
    mode: str = "nearest",
    positive: bool = True,
) -> np.array:
    """Upscales an array by the upscaling factor.

    Parameters
    ----------
    array : np.array
        The array to be upscaled
    upscaling_factor : int
        The upscaling factor
    order : int, optional
        order of the spline polynomial (max 5), by default 3
    mode : str, optional
        extrapolation mode applied at the edges of the array, by default "nearest"
    positive : bool, optional
        set all negative values to 0, by default True

    Returns
    -------
    np.array
        Array that is upscaled by a factor of upscaling_factor
    """
    # upscaling_factor must be integer
    xs = []
    for d in array.shape:
        n_bins = d
        points = np.linspace(
            -0.5 + 1 / 2 / upscaling_factor,
            n_bins - 0.5 - 1 / 2 / upscaling_factor,
            n_bins * upscaling_factor,
        )
        xs.append(points)

    # return the smoothed array
    xy = np.meshgrid(*xs, indexing="ij")
    smoothed = ndimage.map_coordinates(array, xy, order=order, mode=mode)
    if positive:
        smoothed[smoothed < 0] = 0
    return smoothed


def upscale_array_regionally(
    array: np.array,
    upscaling_factor: int,
    num_bins: list,
    order: int = 3,
    mode: str = "nearest",
    positive: bool = True,
) -> np.array:
    """Upscales an array by the upscaling factor separately in each region of the array.

    Parameters
    ----------
    array : np.array
        array to be upscaled
    upscaling_factor : int
        upscaling factor
    num_bins : list
        list of lists of region lengths in each dimension,
        region lengths should sum to the length of the array in that dimension
    order : int, optional
        order of the spline polynomial (max 5), by default 3
    mode : str, optional
        extrapolation mode applied at the edges of each region, by default "nearest"
    positive : bool, optional
        set all negative values to 0, by default True

    Returns
    -------
    np.array
        Array that is upscaled by a factor of upscaling_factor
    """
    up_array = np.empty(shape=[ds * upscaling_factor for ds in array.shape])
    starts = [np.cumsum([0] + regionlengths)[:-1] for regionlengths in num_bins]
    starts_grid = np.meshgrid(*starts)
    starts_grid = [starts_grid[i].flatten() for i in range(len(starts_grid))]
    finishes = [np.cumsum(regionlengths) for regionlengths in num_bins]
    finishes_grid = np.meshgrid(*finishes)
    finishes_grid = [finishes_grid[i].flatten() for i in range(len(finishes_grid))]
    d = len(array.shape)
    for i in range(len(starts_grid[0])):
        slice_bounds = []
        slice_obj_up_bounds = []
        for j in range(d):
            slice_bounds.append(slice(starts_grid[j][i], finishes_grid[j][i]))
            slice_obj_up_bounds.append(
                slice(
                    starts_grid[j][i] * upscaling_factor,
                    finishes_grid[j][i] * upscaling_factor,
                )
            )
        slice_obj = tuple(slice_bounds)
        slice_obj_up = tuple(slice_obj_up_bounds)
        up_array[slice_obj_up] = upscale_array(
            array[slice_obj],
            upscaling_factor,
            order=order,
            mode=mode,
            positive=positive,
        )
    return up_array
