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
    cube_root = np.sign(CM) * np.abs(CM)**(1 / 3) # handles negative weights just in case
    cycles3 = np.diag(np.dot(cube_root, np.dot(cube_root, cube_root)))
    d[np.where(cycles3 == 0)] = np.inf  # set coef to 0 when no 3-cycles exist
    ccoef = cycles3 / (d * (d - 1))
    return ccoef

def transitivity(CM: np.ndarray) -> np.float64:
    """
    Computes transitivity as the ratio of triangles to triplets in the network,
    where a triangle refers to a closed triplet (nodes that are connected by three connections)

    Args:
    ---------------
    CM: NxN np.ndarray, undirected weighted connectivity matrix

    Returns:
    ---------------
    t: np.float64, transitivity ratio

    """
    d = np.sum(np.logical_not(CM == 0), axis=1)
    cube_root = np.sign(CM) * np.abs(CM)**(1 / 3) # handles negative weights just in case
    cycles3 = np.diag(np.dot(cube_root, np.dot(cube_root, cube_root)))
    t = np.sum(cycles3, axis=0) / np.sum(d * (d - 1), axis=0)
    return t

def eigenvalues(CM: np.ndarray) -> np.ndarray:
    """
    Computes the eigenvalues of the connectivity (correlation) matrix CM

    Args:
    ---------------
    CM: NxN np.ndarray, undirected weighted connectivity matrix

    Returns:
    ---------------
    eigen_values: Nx1 np.ndarray, eigenvalues of CM

    """
    eigen_values, eigen_vectors = np.linalg.eig(CM)
    eigen_values = np.absolute(eigen_values)
    return eigen_values

def upper_right_triangle(CM: np.ndarray) -> np.ndarray:
    """
    Extracts the upper right triangle of the correlation coefficients (connectivity matrix CM)
    This includes the diagonal

    Args:
    ---------------
    CM: NxN np.ndarray, undirected weighted connectivity matrix

    Returns:
    ---------------
    corr: ((N*(N+1))/2)x1 np.ndarray, correlation coefficients in the upper right triangle of the matrix CM
          e.g. For a CM of shape 19x19, corr will have a shape of 190x1
          
    """
    mask = np.triu(np.ones(CM.shape, dtype=bool))
    corr = CM[mask]
    return corr