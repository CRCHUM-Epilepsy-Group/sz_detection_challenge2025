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

def clustering_coef(CM: np.ndarray) -> np.ndarray:
    """
    Computes weighted clustering coefficient from weighted undirected connectivity matrix.
    The clustering coefficient is the average intensity of triangles around a node

    Args:
    ---------------
    CM: NxN np.ndarray, undirected weighted connectivity matrix

    Returns:
    ---------------
    ccoef: Nx1 np.ndarray, clustering coefficients

    """
    d = np.array(np.sum(np.logical_not(CM == 0), axis=1), dtype=float) # computes node degree
    cube_root = np.sign(CM) * np.abs(CM)**(1 / 3) # handles negative weights
    cycles3 = np.diag(np.dot(cube_root, np.dot(cube_root, cube_root)))
    d[np.where(cycles3 == 0)] = np.inf  # set coef to 0 when no 3-cycles exist
    ccoef = cycles3 / (d * (d - 1))
    return ccoef