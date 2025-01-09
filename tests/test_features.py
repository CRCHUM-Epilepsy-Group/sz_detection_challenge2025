from pathlib import Path
import numpy as np
from szdetect import centrality_features as cenf
from szdetect import efficiency_features as eff

test_dir = Path(__file__).parent

matrix = np.load('C:/Users/p0121182/Project/challenge/tests/test_data/connectivity_segmented.npy')
slice = matrix[0]

def test_betweenness(CM): 
    assert CM.ndim == 2, f"Expected 2 dimensions, got {CM.ndim}"
    betweenness = cenf.betweenness(CM) 
    print(f'Betweenness shape: {betweenness.shape}')
    return betweenness

def test_diversity_coef(CM): 
    assert CM.ndim == 2, f"Expected 2 dimensions, got {CM.ndim}"
    Hpos = cenf.diversity_coef(CM)
    print(f'Diversity coefficient(positive) shape: {Hpos.shape}')
    return Hpos

def test_node_betweenness(CM):
    assert CM.ndim == 2, f"Expected 2 dimensions, got {CM.ndim}"
    bc = cenf.node_betweenness(CM)
    print(f'Node betweenness shape: {bc.shape}')
    return bc

def test_participation_coef(CM):
    assert CM.ndim == 2, f"Expected 2 dimensions, got {CM.ndim}"
    part_coef = cenf.participation_coef(CM)
    print(f'Participation coefficient shape: {part_coef.shape}')
    return part_coef

def test_module_degree_zscore(CM):
    assert CM.ndim == 2, f"Expected 2 dimensions, got {CM.ndim}"
    zscore = cenf.module_degree_zscore(CM)
    print(f'Module degree zscore shape: {zscore.shape}')
    return zscore

def test_eigenvec_centrality(CM):
    assert CM.ndim == 2, f"Expected 2 dimensions, got {CM.ndim}"
    eigenvec = cenf.eigenvector_centrality(CM)
    print(f'Eigenvector centrality shape: {eigenvec.shape}')
    return eigenvec

def test_efficiency(CM):
    assert CM.ndim == 2, f"Expected 2 dimensions, got {CM.ndim}"
    efficiency = eff.efficiency(CM)
    print(f'Efficiency value: {efficiency}')
    return efficiency

def test_global_diffusion_efficiency(CM):
    assert CM.ndim == 2, f"Expected 2 dimensions, got {CM.ndim}"
    gediff = eff.global_diffusion_efficiency(CM)
    print(f'Mean global diffusion efficiency: {gediff}')
    return gediff


def test_global_rout_efficiency(CM):
    assert CM.ndim == 2, f"Expected 2 dimensions, got {CM.ndim}"
    GErout = eff.global_rout_efficiency(CM)
    print(f'Mean global routing efficiency: {GErout}')
    return GErout

def test_local_rout_efficiency(CM):
    assert CM.ndim == 2, f"Expected 2 dimensions, got {CM.ndim}"
    Eloc = eff.local_rout_efficiency(CM)
    print(f'Local efficiency shape: {Eloc.shape}')
    return Eloc


betweenness = test_betweenness(slice)
Hpos = test_diversity_coef(slice)
bc = test_node_betweenness(slice)
part_coef = test_participation_coef(slice)
zscore = test_module_degree_zscore(slice)
eigenvec = test_eigenvec_centrality(slice)
efficiency = test_efficiency(slice)
gediff = test_global_diffusion_efficiency(slice)
GErout = test_global_rout_efficiency(slice)
Eloc = test_local_rout_efficiency(slice)

