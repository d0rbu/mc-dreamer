import os
import yaml
import numpy as np
import pytest
import torch as th
from itertools import product
from typing import Sequence, Callable
from core.extract.filters.natural_convolution import natural_convolution_filter
from core.extract.filters.solid_blocks import solid_block_filter

NUM_BLOCK_TYPES = 256


### TESTS ###
def test_solid_block_filter_1():
    # arrange
    test_input = th.tensor([[[0, 12, 0], [14, 10, 11], [0, 77, 0]]])
    expected = th.tensor([[[False, True, False], [True, True, True], [False, True, False]]])

    # act
    result = solid_block_filter(test_input, None)

    # assert
    assert th.all(result == expected)

def test_solid_block_filter_2():
    # arrange
    test_input = th.tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    expected = th.tensor([[[False, False, False], [False, False, False], [False, False, False]]])
    
    # act
    result = solid_block_filter(test_input, None)

    # assert
    assert th.all(result == expected)

def test_solid_block_filter_3():
    # arrange
    test_input = th.tensor([[[0, 12, 0], [14, 10, 11], [0, 77, 0]]])
    test_mask = th.tensor([[[True, True, False], [True, True, True], [False, False, False]]])
    expected = th.tensor([[[False, True, False], [True, True, True], [False, True, False]]])

    # act
    result = solid_block_filter(test_input, test_mask)

    # assert
    assert th.all(result[test_mask] == expected[test_mask])

def test_natural_convolution_filter_1():
    # arrange
    test_input = th.full((512, 256, 512), 210) # No natural blocks
    test_mask =th.full((512, 256, 512), 1) 
    expected = th.full((512, 256, 512), 1) # So all true

    # act
    result = solid_block_filter(test_input, test_mask)

    # assert
    assert th.all(result[test_mask] == expected[test_mask])

def test_natural_convolution_filter_2():
    # arrange (threshold test)
    test_input = th.full((512, 256, 512), 210)
    test_input[1,1,1:2] = 1
    test_input[1,65,64] = 1
    test_input[198,71:73,200] = 1
    test_input[412:413,200,234:235] = 1 # These should not be enough to trigger a false (i think)
    test_mask =th.full((512, 256, 512), 1) 
    expected = th.full((512, 256, 512), 1) # So all true

    # act
    result = solid_block_filter(test_input, test_mask)

    # assert
    assert th.all(result[test_mask] == expected[test_mask])


def test_natural_convolution_filter_3():
    # arrange 
    test_input = th.full((512, 256, 512), 210) # Making checkered 3d diamond at the center
    dims = (512, 256, 512) 
    center = (dims[0] // 2, dims[1] // 2, dims[2] // 2) # Calculate the center
    square_radius = 60
    # Populate the tensor with 1s inside the diamond
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                distance = abs(x - center[0]) + abs(y - center[1]) + abs(z - center[2])
                if distance <= square_radius:
                    test_input[x, y, z] = (x + y + z) % 2

    test_mask =th.full((512, 256, 512), 1) 

    expected = th.full((512, 256, 512), 1) # Making checkered 3d diamond at the center
    dims = (512, 256, 512) 
    center = (dims[0] // 2, dims[1] // 2, dims[2] // 2) # Calculate the center
    shorter_square_radius = 59 # THIS IS PROBABLY NOT RIGHT
    
    # Populate the tensor with 1s inside the diamond
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                distance = abs(x - center[0]) + abs(y - center[1]) + abs(z - center[2])
                if distance <= shorter_square_radius:
                    test_input[x, y, z] = 0

    # act
    result = solid_block_filter(test_input, test_mask)

    # assert
    assert th.all(result[test_mask] == expected[test_mask])

