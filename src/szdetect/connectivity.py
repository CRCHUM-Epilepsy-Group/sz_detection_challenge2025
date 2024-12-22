import numpy as np

def node_degree(CM: np.ndarray) -> np.ndarray:
    """
    Computes binary node degree from undirected connectivity matrix
    (to analyze connectivity presence). 

    Args:
    ---------------
    CM: NxN np.ndarray, undirected connectivity matrix

    Returns:
    ---------------
    ndeg: Nx1 np.ndarray, node degree

    """

    CM = CM.copy()
    CM[CM != 0] = 1   # binarizes conn matrix
    ndeg = np.sum(CM, axis=0)
    return ndeg

def node_strength(CM: np.ndarray) -> np.ndarray:
    # TODO: use signed approach and return dictionary?
    """
    Computes weighted node degree (node strengths) from undirected connectivity matrix

    Args:
    ---------------
    CM: NxN np.ndarray, undirected weighted connectivity matrix

    Returns:
    ---------------
    strg: Nx1 np.ndarray, node strengths

    """
    strg = np.sum(CM, axis=0)
    return strg