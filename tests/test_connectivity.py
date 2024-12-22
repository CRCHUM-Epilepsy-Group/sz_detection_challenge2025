#import pytest
from pathlib import Path
import numpy as np
from szdetect import connectivity

test_dir = Path(__file__).parent
CM = np.load(test_dir / 'test_data/connectivity_segmented.npy')

def test_node_degree(CM): 
    assert CM.ndim == 2, f"Expected 2 dimensions, got {CM.ndim}"   
    ndeg = connectivity.node_degree(CM) 
    #print(ndeg)
    return ndeg

def test_node_strength(CM): 
    assert CM.ndim == 2, f"Expected 2 dimensions, got {CM.ndim}"   
    strg = connectivity.node_strength(CM) 
    #print(strg)
    return strg

ndeg_list = [test_node_degree(m) for m in CM] 
print(f"Length of ndeg_list: {len(ndeg_list)}")
print(f"Type of items in list: {type(ndeg_list[0])}")
print(f"Shape of items in list: {ndeg_list[0].shape}")
# TODO: is this implementation correct? There are no zeros,
# so of course the nodal degree will always be 18 for a 19x19 CM,
# meaning that each node is connected to all other 18 nodes

strg_list = [test_node_strength(m) for m in CM]
print(f"Length of strg_list: {len(strg_list)}")
print(f"Type of items in list: {type(strg_list[0])}")
print(f"Shape of items in list: {strg_list[0].shape}")