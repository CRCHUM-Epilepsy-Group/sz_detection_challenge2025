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
    Hpos, Hneg = cenf.diversity_coef(CM)
    print(f'Diversity coefficient(positive) shape: {Hpos.shape}')
    print(f'Diversity coefficient(negative) shape: {Hneg.shape}')
    return Hpos, Hneg

def test_edge_betweenness(CM):
    assert CM.ndim == 2, f"Expected 2 dimensions, got {CM.ndim}"
    ebc, bc = cenf.edge_betweenness(CM)
    print(f'Edge vertex betweenness shape: {ebc.shape}')
    print(f'Node betweenness shape: {bc.shape}')
    return ebc, bc

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
    eigenvec = cenf.eigenvec_centrality(CM)
    print(f'Eigenvector centrality shape: {eigenvec.shape}')
    return eigenvec

def test_efficiency(CM):
    assert CM.ndim == 2, f"Expected 2 dimensions, got {CM.ndim}"
    efficiency = eff.efficiency(CM)
    print(f'Efficiency shape: {efficiency.shape}')
    return efficiency

def test_diffusion_efficiency(CM):
    assert CM.ndim == 2, f"Expected 2 dimensions, got {CM.ndim}"
    gediff, ediif = eff.diffusion_efficiency(CM)
    print(f'mean global diffusion efficiency shape: {gediff.shape}')
    print(f'pairwise diffusion efficiency shape: {ediif.shape}')
    return gediff, ediif

def test_rout_efficiency(CM):
    assert CM.ndim == 2, f"Expected 2 dimensions, got {CM.ndim}"
    GErout, Erout, Eloc = eff.rout_efficiency(CM)
    print(f'Mean global routing eff shape: {GErout.shape}')
    print(f'Pairwise routing eff shape: {Erout.shape}')
    print(f'Local efficiency shape: {Eloc.shape}')
    return GErout, Erout, Eloc

values_centrality = test_betweenness(slice)
print(values_centrality)

value_eff1, _ = test_diffusion_efficiency(slice)
print(value_eff1)