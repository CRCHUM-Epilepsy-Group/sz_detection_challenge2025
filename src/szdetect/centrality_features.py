import numpy as np
import networkx as nx
import community as community_louvain


def betweenness(G):
    '''
    Node betweenness centrality is the fraction of all shortest paths in
    the network that contain a given node. Nodes with high values of
    betweenness centrality participate in a large number of shortest paths.

    Parameters:
    L (NxN np.ndarray) : directed/undirected weighted connection matrix

    Returns:
    BC (Nx1 np.ndarray) : node betweenness centrality vector

    '''
    n = len(G)
    BC = np.zeros((n,))  # vertex betweenness

    for u in range(n):
        D = np.tile(np.inf, (n,))
        D[u] = 0  # distance from u
        NP = np.zeros((n,))
        NP[u] = 1  # number of paths from u
        S = np.ones((n,), dtype=bool)  # distance permanence
        P = np.zeros((n, n))  # predecessors
        Q = np.zeros((n,), dtype=int)  # indices
        q = n - 1  # order of non-increasing distance

        G1 = G.copy()
        V = [u]
        while True:
            S[V] = 0  # distance u->V is now permanent
            G1[:, V] = 0  # no in-edges as already shortest
            for v in V:
                Q[q] = v
                q -= 1
                W, = np.where(G1[v, :])  # neighbors of v
                for w in W:
                    Duw = D[v] + G1[v, w]  # path length to be tested
                    if Duw < D[w]:  # if new u->w shorter than old
                        D[w] = Duw
                        NP[w] = NP[v]  # NP(u->w) = NP of new path
                        P[w, :] = 0
                        P[w, v] = 1  # v is the only predecessor
                    elif Duw == D[w]:  # if new u->w equal to old
                        NP[w] += NP[v]  # NP(u->w) sum of old and new
                        P[w, v] = 1  # v is also predecessor

            if D[S].size == 0:
                break  # all nodes were reached
            if np.isinf(np.min(D[S])):  # some nodes cannot be reached
                Q[:q + 1], = np.where(np.isinf(D))  # these are first in line
                break
            V, = np.where(D == np.min(D[S]))

        DP = np.zeros((n,))
        for w in Q[:n - 1]:
            BC[w] += DP[w]
            for v in np.where(P[w, :])[0]:
                DP[v] += (1 + DP[w]) * NP[v] / NP[w]

    return BC


def diversity_coef(W):
    '''
    Calculates the Shannon-entropy based diversity coefficient, which measures the
    diversity of intermodular connections of individual nodes, ranging from 0 to 1.

    Parameters :
    W (NxN np.ndarray): Undirected connection matrix

    Returns :
    Hpos (Nx1 np.ndarray): Diversity coefficient based on positive connections
    Hneg (Nx1 np.ndarray): Diversity coefficient based on negative connections
    '''
    # Create a graph
    G = nx.from_numpy_array(W)

    # Detect communities
    partition = community_louvain.best_partition(G)

    # Convert partition to a numpy array (ci vector)
    ci = np.array([partition[node] for node in G.nodes()])
    # Relabeling communities for easy indexing
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    print(ci)
    print(ci.shape)
    n = len(W)  # number of nodes
    m = np.max(ci)  # number of modules

    # Function to calculate entropy
    def entropy(w):
        S = np.sum(w, axis=1)  # strength
        Snm = np.zeros((n, m))  # node-to-module degree
        for i in range(m):
            Snm[:, i] = np.sum(w[:, ci == (i + 1)], axis=1)
        pnm = Snm / (np.tile(S, (m, 1)).T)
        pnm[np.isnan(pnm)] = 0  # Handle division by zero
        return -np.sum(pnm * np.log2(np.clip(pnm, 1e-10, 1)), axis=1) / np.log2(m)

    # Handling errors explicitly
    with np.errstate(invalid='ignore', divide='ignore'):
        Hpos = entropy(W * (W > 0))
        Hneg = entropy(-W * (W < 0))

    return Hpos, Hneg


def edge_betweenness(G):
    '''
    Edge betweenness centrality is the fraction of all shortest paths in
    the network that contain a given edge. Edges with high values of
    betweenness centrality participate in a large number of shortest paths.

    Parameters :
    L (NxN np.ndarray) : directed/undirected weighted connection matrix

    Returns :
    EBC (NxN np.ndarray) : edge betweenness centrality matrix
    BC (Nx1 np.ndarray) : node betweenness centrality vector
    '''
    n = len(G)
    BC = np.zeros((n,))  # vertex betweenness
    EBC = np.zeros((n, n))  # edge betweenness

    for u in range(n):
        D = np.tile(np.inf, n)
        D[u] = 0  # distance from u
        NP = np.zeros((n,))
        NP[u] = 1  # number of paths from u
        S = np.ones((n,), dtype=bool)  # distance permanence
        P = np.zeros((n, n))  # predecessors
        Q = np.zeros((n,), dtype=int)  # indices
        q = n - 1  # order of non-increasing distance

        G1 = G.copy()
        V = [u]
        while True:
            S[V] = 0  # distance u->V is now permanent
            G1[:, V] = 0  # no in-edges as already shortest
            for v in V:
                Q[q] = v
                q -= 1
                W, = np.where(G1[v, :])  # neighbors of v
                for w in W:
                    Duw = D[v] + G1[v, w]  # path length to be tested
                    if Duw < D[w]:  # if new u->w shorter than old
                        D[w] = Duw
                        NP[w] = NP[v]  # NP(u->w) = NP of new path
                        P[w, :] = 0
                        P[w, v] = 1  # v is the only predecessor
                    elif Duw == D[w]:  # if new u->w equal to old
                        NP[w] += NP[v]  # NP(u->w) sum of old and new
                        P[w, v] = 1  # v is also a predecessor

            if D[S].size == 0:
                break  # all nodes reached, or
            if np.isinf(np.min(D[S])):  # some cannot be reached
                Q[:q + 1], = np.where(np.isinf(D)) # these are first in line.
                break
            V, = np.where(D == np.min(D[S]))

        DP = np.zeros((n,))  # dependency
        for w in Q[:n - 1]:
            BC[w] += DP[w]
            for v in np.where(P[w, :])[0]:
                DPvw = (1 + DP[w]) * NP[v] / NP[w]
                DP[v] += DPvw
                EBC[v, w] += DPvw

    return EBC, BC

def participation_coef(W):
    '''
    Participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.  

    Parameters :
    W (NxN np.ndarray) : binary/weighted directed/undirected connection matrix

    Returns :
    P (Nx1 np.ndarray) : participation coefficient
    '''
    G = nx.from_numpy_array(W)

    # Detect communities
    partition = community_louvain.best_partition(G)

    # Convert partition to a numpy array (ci vector)
    ci = np.array([partition[node] for node in G.nodes()])
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    n = len(W)  # number of vertices
    Ko = np.sum(W, axis=1)  # (out) degree
    Gc = np.dot((W != 0), np.diag(ci))  # neighbor community affiliation
    Kc2 = np.zeros((n,))  # community-specific neighbors

    for i in range(1, int(np.max(ci)) + 1):
        Kc2 = Kc2 + np.square(np.sum(W * (Gc == i), axis=1))

    P = np.ones((n,)) - Kc2 / np.square(Ko)
    # P=0 if for nodes with no (out) neighbors
    P[np.where(np.logical_not(Ko))] = 0

    return P

def module_degree_zscore(W):
    '''
    The within-module degree z-score is a within-module version of degree
    centrality.

    Parameters :
    W (NxN np.narray) : binary/weighted directed/undirected connection matrix

    Returns :
    Z (Nx1 np.ndarray) : within-module degree Z-score
    '''
    G = nx.from_numpy_array(W)

    # Detect communities
    partition = community_louvain.best_partition(G)

    # Convert partition to a numpy array (ci vector)
    ci = np.array([partition[node] for node in G.nodes()])
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    n = len(W)
    Z = np.zeros((n,))  # number of vertices
    for i in range(1, int(np.max(ci) + 1)):
        Koi = np.sum(W[np.ix_(ci == i, ci == i)], axis=1)
        Z[np.where(ci == i)] = (Koi - np.mean(Koi)) / np.std(Koi)

    Z[np.where(np.isnan(Z))] = 0
    return Z

def eigenvector_centrality(CIJ):
    from scipy import linalg
    n = len(CIJ)
    vals, vecs = linalg.eig(CIJ)
    i = np.argmax(vals)
    return np.abs(vecs[:, i])