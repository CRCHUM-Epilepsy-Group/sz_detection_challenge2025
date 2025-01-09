import numpy as np


############################################################################################################
#Support functions¸

def cuberoot(x):
    """
    Correctly handles the cube root for all weights.

    """
    return np.sign(x) * np.abs(x) ** (1 / 3)

def invert(CM: np.ndarray, copy: bool = True) -> np.ndarray:
    """
    Inverts elementwise the weights in a connection matrix.

    Args:
    ---------------
    CM: NxN np.ndarray, connection matrix
    copy: bool, whether to operate on a copy of the matrix

    Returns:
    ---------------
    inverted_CM: NxN np.ndarray, inverted connection matrix

    """
    if copy:
        CM = CM.copy()
    indices = np.where(CM)
    CM[indices] = 1.0 / CM[indices]
    return CM

def distance_wei_floyd(CM: np.ndarray, transform: str = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the shortest path lengths between all node pairs using the Floyd-Warshall algorithm.

    Args:
    ---------------
    CM: NxN np.ndarray, connection matrix
    transform: str, optional transformation ('log' or 'inv') applied to weights

    Returns:
    ---------------
    SPL: NxN np.ndarray, shortest path lengths
    hops: NxN np.ndarray, number of hops
    Pmat: NxN np.ndarray, predecessor matrix

    """
    if transform is not None:
        with np.errstate(divide='ignore'):
            if transform == 'log':
                SPL = -np.log(CM)
            elif transform == 'inv':
                SPL = 1 / CM
            else:
                raise ValueError("Unexpected transform type. Only 'log' and 'inv' are accepted.")
    else:
        SPL = CM.copy().astype('float')
        SPL[SPL == 0] = np.inf

    n = CM.shape[1]
    hops = np.array(CM != 0).astype('float')
    Pmat = np.repeat(np.atleast_2d(np.arange(0, n)), n, 0)

    for k in range(n):
        i2k_k2j = np.repeat(SPL[:, [k]], n, 1) + np.repeat(SPL[[k], :], n, 0)
        path = SPL > i2k_k2j
        i, j = np.where(path)
        hops[path] = hops[i, k] + hops[k, j]
        Pmat[path] = Pmat[i, k]
        SPL = np.minimum(SPL, i2k_k2j)

    np.fill_diagonal(SPL, 0)
    hops[np.eye(n, dtype=bool)] = 0
    Pmat[np.eye(n, dtype=bool)] = 0

    return SPL, hops, Pmat

def mean_first_passage_time(CM: np.ndarray) -> np.ndarray:
    """
    Calculates the mean first passage time of a connection matrix.

    Args:
    ---------------
    CM: NxN np.ndarray, connection matrix

    Returns:
    ---------------
    MFPT: NxN np.ndarray, pairwise mean first passage time array

    """
    P = np.linalg.solve(np.diag(np.sum(CM, axis=1)), CM)
    n = len(P)
    D, V = np.linalg.eig(P.T)

    aux = np.abs(D - 1)
    index = np.where(aux == aux.min())[0]

    if aux[index] > 1e-3:
        raise ValueError(f"Cannot find eigenvalue of 1. Minimum eigenvalue value is {aux[index]}. Tolerance set at 1e-3.")

    w = V[:, index].T
    w /= np.sum(w)

    W = np.real(np.repeat(w, n, 0))
    I = np.eye(n)
    Z = np.linalg.inv(I - P + W)
    mfpt = (np.repeat(np.atleast_2d(np.diag(Z)), n, 0) - Z) / W

    return mfpt



############################################################################################################
# Features for efficiency

def efficiency(CM: np.ndarray, local: bool = False) -> float:
    """
    Calculates global or local efficiency of a graph.

    Args:
    ---------------
    CM: NxN np.ndarray, undirected weighted connection matrix
    local: bool, calculate local efficiency if True

    Returns:
    ---------------
    Eglob: float, global efficiency if local is False

    """
    local = False
    def distance_inv_wei(G):
        n = len(G)
        D = np.zeros((n, n))  # distance matrix
        D[np.logical_not(np.eye(n))] = np.inf

        for u in range(n):
            # distance permanence (true is temporary)
            S = np.ones((n,), dtype=bool)
            G1 = G.copy()
            V = [u]
            while True:
                S[V] = 0  # distance u->V is now permanent
                G1[:, V] = 0  # no in-edges as already shortest
                for v in V:
                    W, = np.where(G1[v, :])  # neighbors of smallest nodes
                    td = np.array(
                        [D[u, W].flatten(), (D[u, v] + G1[v, W]).flatten()])
                    D[u, W] = np.min(td, axis=0)

                if D[u, S].size == 0:  # all nodes reached
                    break
                minD = np.min(D[u, S])
                if np.isinf(minD):  # some nodes cannot be reached
                    break
                V, = np.where(D[u, :] == minD)

        np.fill_diagonal(D, 1)
        D = 1 / D
        np.fill_diagonal(D, 0)
        return D

    n = len(CM)
    Gl = invert(CM, copy=True)  # connection length matrix
    A = np.array((CM != 0), dtype=int)
    #local efficiency algorithm described by Rubinov and Sporns 2010, not recommended
    if local == 'original':
        E = np.zeros((n,))
        for u in range(n):
            # find pairs of neighbors
            V, = np.where(np.logical_or(CM[u, :], CM[:, u].T))
            # symmetrized vector of weights
            sw = cuberoot(CM[u, V]) + cuberoot(CM[V, u].T)
            # inverse distance matrix
            e = distance_inv_wei(Gl[np.ix_(V, V)])
            # symmetrized inverse distance matrix
            se = cuberoot(e) + cuberoot(e.T)

            numer = np.sum(np.outer(sw.T, sw) * se) / 2
            if numer != 0:
                # symmetrized adjacency vector
                sa = A[u, V] + A[V, u].T
                denom = np.sum(sa)**2 - np.sum(sa * sa)
                # print numer,denom
                E[u] = numer / denom  # local efficiency

    #local efficiency algorithm described by Wang et al 2016, recommended
    elif local in (True, 'local'):
        E = np.zeros((n,))
        for u in range(n):
            V, = np.where(np.logical_or(CM[u, :], CM[:, u].T))
            sw = cuberoot(CM[u, V]) + cuberoot(CM[V, u].T)
            e = distance_inv_wei(cuberoot(Gl)[np.ix_(V, V)])
            se = e+e.T
         
            numer = np.sum(np.outer(sw.T, sw) * se) / 2
            if numer != 0:
                # symmetrized adjacency vector
                sa = A[u, V] + A[V, u].T
                denom = np.sum(sa)**2 - np.sum(sa * sa)
                # print numer,denom
                E[u] = numer / denom  # local efficiency

    elif local in (False, 'global'):
        e = distance_inv_wei(Gl)
        E = np.sum(e) / (n * n - n)
    return E



def global_diffusion_efficiency(CM: np.ndarray) -> float:
    """
    Calculates the efficiency of information diffusion in the network.

    Args:
    ---------------
    CM: NxN np.ndarray, connection matrix

    Returns:
    ---------------
    gediff: float, mean global diffusion efficiency

    """
    n = len(CM)
    CM_copy = CM.copy()
    mfpt = mean_first_passage_time(CM_copy)
    with np.errstate(divide='ignore'):
        ediff = 1 / mfpt
    np.fill_diagonal(ediff, 0)
    gediff = np.sum(ediff) / (n ** 2 - n)

    return gediff


def global_rout_efficiency(CM: np.ndarray) -> float:
    """
    Calculates routing efficiency of the network.

    Args:
    ---------------
    CM: NxN np.ndarray, connection matrix

    Returns:
    ---------------
    GErout: float, mean global routing efficiency
    """
    n = len(CM)
    Erout, _, _ = distance_wei_floyd(CM, transform=None)
    with np.errstate(divide='ignore'):
        Erout = 1 / Erout
    np.fill_diagonal(Erout, 0)
    GErout = (np.sum(Erout[np.where(np.logical_not(np.isnan(Erout)))]) / 
              (n ** 2 - n))
    
    return GErout

def local_rout_efficiency(CM: np.ndarray) -> np.ndarray:
    """
    Calculates routing efficiency of the network.

    Args:
    ---------------
    CM: NxN np.ndarray, connection matrix

    Returns:
    ----------------
    Eloc: Nx1 np.ndarray, local routing efficiency vector

    """
    n = len(CM)
    Erout, _, _ = distance_wei_floyd(CM, transform=None)
    with np.errstate(divide='ignore'):
        Erout = 1 / Erout
    np.fill_diagonal(Erout, 0)

    Eloc = np.zeros((n,))
    for u in range(n):
        Gu, = np.where(np.logical_or(CM[u, :], CM[:, u].T))
        nGu = len(Gu)
        e, _, _ = distance_wei_floyd(CM[Gu, :][:, Gu], transform=None)
        with np.errstate(divide='ignore'):
            e = 1 / e
        np.fill_diagonal(e, 0)
        Eloc[u] = np.sum(e) / nGu

    return Eloc