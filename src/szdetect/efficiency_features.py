import numpy as np

def cuberoot(x):
        '''
        Correctly handle the cube root for all weights
        '''
        return np.sign(x) * np.abs(x)**(1 / 3)

def invert(W, copy=True):
    '''
    Inverts elementwise the weights in an input connection matrix.
    '''
    if copy:
        W = W.copy()
    E = np.where(W)
    W[E] = 1. / W[E]
    return W

def distance_wei_floyd(adjacency, transform=None):
    """
    Computes the topological length of the shortest possible path connecting 
    every pair of nodes in the network

    """

    #it is important not to do these transformations safely, to allow infinity
    if transform is not None:
        with np.errstate(divide='ignore'):
            if transform == 'log':
                #SPL = logtransform(adjacency)
                SPL = -np.log(adjacency)
            elif transform == 'inv':
                #SPL = invert(adjacency)
                SPL = 1 / adjacency
            else:
                raise ValueError("Unexpected transform type. Only 'log' and " +
                                 "'inv' are accepted")
    else:
        SPL = adjacency.copy().astype('float')
        SPL[SPL == 0] = np.inf

    n = adjacency.shape[1]

    hops = np.array(adjacency != 0).astype('float')
    Pmat = np.repeat(np.atleast_2d(np.arange(0, n)), n, 0)

    #print(SPL)

    for k in range(n):
        i2k_k2j = np.repeat(SPL[:, [k]], n, 1) + np.repeat(SPL[[k], :], n, 0)

        path = SPL > i2k_k2j
        i, j = np.where(path)
        hops[path] = hops[i, k] + hops[k, j]
        Pmat[path] = Pmat[i, k]

        SPL = np.min(np.stack([SPL, i2k_k2j], 2), 2)

    I = np.eye(n) > 0
    SPL[I] = 0

    hops[I], Pmat[I] = 0, 0

    return SPL, hops, Pmat

def mean_first_passage_time(adjacency):
    """
    Calculates mean first passage time of adjacency

    Parameters :
    adjacency (NxN array_like) : Weighted/unweighted, direct/undirected connection

    Returns :
    MFPT (NxN ndarray) : Pairwise mean first passage time array
    """

    P = np.linalg.solve(np.diag(np.sum(adjacency, axis=1)), adjacency)

    n = len(P)
    D, V = np.linalg.eig(P.T)

    aux = np.abs(D - 1)
    index = np.where(aux == aux.min())[0]

    if aux[index] > 10e-3:
        raise ValueError("Cannot find eigenvalue of 1. Minimum eigenvalue " +
                         "value is {0}. Tolerance was ".format(aux[index]+1) +
                         "set at 10e-3.")

    w = V[:, index].T
    w = w / np.sum(w)

    W = np.real(np.repeat(w, n, 0))
    I = np.eye(n)

    Z = np.linalg.inv(I - P + W)

    mfpt = (np.repeat(np.atleast_2d(np.diag(Z)), n, 0) - Z) / W

    return mfpt

def efficiency(Gw, local = False):
    '''
    Calculate the global efficiency of a graph.

    Parameters :
    Gw (NxN np.ndarray) : Undirected weighted connection matrix where each element represents 
    the weight of the edge between nodes. 
       
    Returns :
    Eglob (float) :  Global efficiency of the graph, calculated as the average of the inverse 
    shortest path lengths between all pairs of nodes.

    '''
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

    n = len(Gw)
    Gl = invert(Gw, copy=True)  # connection length matrix
    A = np.array((Gw != 0), dtype=int)
    #local efficiency algorithm described by Rubinov and Sporns 2010, not recommended
    if local == 'original':
        E = np.zeros((n,))
        for u in range(n):
            # V,=np.where(Gw[u,:])		#neighbors
            # k=len(V)					#degree
            # if k>=2:					#degree must be at least 2
            #	e=(distance_inv_wei(Gl[V].T[V])*np.outer(Gw[V,u],Gw[u,V]))**1/3
            #	E[u]=np.sum(e)/(k*k-k)

            # find pairs of neighbors
            V, = np.where(np.logical_or(Gw[u, :], Gw[:, u].T))
            # symmetrized vector of weights
            sw = cuberoot(Gw[u, V]) + cuberoot(Gw[V, u].T)
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
            V, = np.where(np.logical_or(Gw[u, :], Gw[:, u].T))
            sw = cuberoot(Gw[u, V]) + cuberoot(Gw[V, u].T)
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


def diffusion_efficiency(adj):
    '''
    Calculate the efficiency of information diffusion across a network.
    The function computes two efficiencies: the pairwise diffusion efficiency between 
    each node pair, and the mean global diffusion efficiency across the network. 

    Parameters:
    adj (NxN np.ndarray) : weighted/unweighted, directed/undirected adjacency matrix

    Returns:
    gediff (float) : mean global diffusion efficiency
    ediff (NxN np.ndarray) : pairwise  diffusion efficiency matrix
    '''
    n = len(adj)
    adj = adj.copy()
    mfpt = mean_first_passage_time(adj)
    with np.errstate(divide='ignore'):
        ediff = 1 / mfpt
    np.fill_diagonal(ediff, 0)
    gediff = np.sum(ediff) / (n ** 2 - n)
    return gediff, ediff

def rout_efficiency(D):
    '''
    The routing efficiency is the average of inverse shortest path length.
 
    The local routing efficiency of a node u is the routing efficiency
    computed on the subgraph formed by the neighborhood of node u (excluding node u).

    Parameters :
    D (NxN np.ndarray) : Weighted/unweighted directed/undirected connection weight or length
    matrix

    Returns :
    GErout (float) : Mean global routing efficiency
    Erout (NxN np.ndarray) : Pairwise routing efficiency matrix
    Eloc (Nx1 np.ndarray) : Local efficiency vector
    '''
    n = len(D)
    Erout, _, _ = distance_wei_floyd(D, transform=None)
    with np.errstate(divide='ignore'):
        Erout = 1 / Erout
    np.fill_diagonal(Erout, 0)
    GErout = (np.sum(Erout[np.where(np.logical_not(np.isnan(Erout)))]) / 
              (n ** 2 - n))

    Eloc = np.zeros((n,))
    for u in range(n):
        Gu, = np.where(np.logical_or(D[u, :], D[:, u].T))
        nGu = len(Gu)
        e, _, _ = distance_wei_floyd(D[Gu, :][:, Gu], transform=None)
        with np.errstate(divide='ignore'):
            e = 1 / e
        np.fill_diagonal(e, 0)
        Eloc[u] = np.sum(e) / nGu

    return GErout, Erout, Eloc