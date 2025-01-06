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

def test_clustering_coef(CM):
    assert CM.ndim == 2, f"Expected 2 dimensions, got {CM.ndim}"   
    ccoef = connectivity.clustering_coef(CM) 
    #print(ccoef)
    return ccoef

def test_transitivity(CM):
    assert CM.ndim == 2, f"Expected 2 dimensions, got {CM.ndim}"   
    t = connectivity.transitivity(CM) 
    #print(t)
    return t

def test_eigenvalues(CM):
    assert CM.ndim == 2, f"Expected 2 dimensions, got {CM.ndim}"   
    eigen_values = connectivity.eigenvalues(CM) 
    #print(eigen_values)
    return eigen_values

def test_upper_right_triangle(CM):
    assert CM.ndim == 2, f"Expected 2 dimensions, got {CM.ndim}"   
    corr = connectivity.upper_right_triangle(CM) 
    #print(corr)
    return corr

ndeg_list = [test_node_degree(m) for m in CM] 
print(f"Length of ndeg_list: {len(ndeg_list)}")
print(f"Type of items in list: {type(ndeg_list[0])}")
print(f"Shape of items in list: {ndeg_list[0].shape}")
# NOTE: not sure if this implementation is correct or maybe the
# feature is not very important? 
# Using the example conn matrices of shape 19x19,
# I always get the same result: an 19x1 array of 18s. 
# Is this expected? Since all values are non-zero except for the diagonal

strg_list = [test_node_strength(m) for m in CM]
print(f"Length of strg_list: {len(strg_list)}")
print(f"Type of items in list: {type(strg_list[0])}")
print(f"Shape of items in list: {strg_list[0].shape}")

ccoef_list = [test_clustering_coef(m) for m in CM] 
print(f"Length of ccoef_list: {len(ccoef_list)}")
print(f"Type of items in list: {type(ccoef_list[0])}")
print(f"Shape of items in list: {ccoef_list[0].shape}")

t_list = [test_transitivity(m) for m in CM] 
print(f"Length of t_list: {len(t_list)}")
print(f"Type of items in list: {type(t_list[0])}")
print(f"Shape of items in list: {t_list[0].shape}")

eigenvalue_list = [test_eigenvalues(m) for m in CM]
print(f"Length of eigenvalue_list: {len(eigenvalue_list)}")
print(f"Type of items in list: {type(eigenvalue_list[0])}")
print(f"Shape of items in list: {eigenvalue_list[0].shape}")

corr_list = [test_upper_right_triangle(m) for m in CM]
print(f"Length of eigenvalue_list: {len(corr_list)}")
print(f"Type of items in list: {type(corr_list[0])}")
print(f"Shape of items in list: {corr_list[0].shape}")

