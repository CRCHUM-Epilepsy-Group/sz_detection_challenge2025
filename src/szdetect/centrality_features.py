import numpy as np
import networkx as nx
import community as community_louvain

############################################################################################################
# features for centrality

def betweenness(CM: np.ndarray) -> np.ndarray:
    """
    Computes node betweenness centrality from a connectivity matrix.

    Args:
    ---------------
    CM: NxN np.ndarray, directed/undirected weighted connection matrix

    Returns:
    ---------------
    BC: Nx1 np.ndarray, node betweenness centrality vector

    """
    n = len(CM)
    BC = np.zeros((n,))

    for u in range(n):
        D = np.tile(np.inf, (n,))
        D[u] = 0  # distance from u
        NP = np.zeros((n,))
        NP[u] = 1  # number of paths from u
        S = np.ones((n,), dtype=bool)  # distance permanence
        P = np.zeros((n, n))  # predecessors
        Q = np.zeros((n,), dtype=int)  # indices
        q = n - 1  # order of non-increasing distance

        CM_temp = CM.copy()
        V = [u]
        while True:
            S[V] = 0  # distance u->V is now permanent
            CM_temp[:, V] = 0  # no in-edges as already shortest
            for v in V:
                Q[q] = v
                q -= 1
                W, = np.where(CM_temp[v, :])  # neighbors of v
                for w in W:
                    Duw = D[v] + CM_temp[v, w]  # path length to be tested
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


def diversity_coef(CM: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the diversity coefficient for nodes based on positive and negative connections.

    Args:
    ---------------
    CM: NxN np.ndarray, undirected connection matrix

    Returns:
    ---------------
    Hpos: Nx1 np.ndarray, diversity coefficient based on positive connections
    Hneg: Nx1 np.ndarray, diversity coefficient based on negative connections

    """
    G = nx.from_numpy_array(CM)
    partition = community_louvain.best_partition(G)
    ci = np.array([partition[node] for node in G.nodes()])
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    n = len(CM)  # number of nodes
    m = np.max(ci)  # number of modules

    def entropy(w):
        S = np.sum(w, axis=1)  # strength
        Snm = np.zeros((n, m))  # node-to-module degree
        for i in range(m):
            Snm[:, i] = np.sum(w[:, ci == (i + 1)], axis=1)
        pnm = Snm / (np.tile(S, (m, 1)).T)
        pnm[np.isnan(pnm)] = 0  # Handle division by zero
        return -np.sum(pnm * np.log2(np.clip(pnm, 1e-10, 1)), axis=1) / np.log2(m)

    with np.errstate(invalid='ignore', divide='ignore'):
        Hpos = entropy(CM * (CM > 0))
        Hneg = entropy(-CM * (CM < 0))

    return Hpos, Hneg


def edge_betweenness(CM: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes edge and node betweenness centrality.

    Args:
    ---------------
    CM: NxN np.ndarray, directed/undirected weighted connection matrix

    Returns:
    ---------------
    EBC: NxN np.ndarray, edge betweenness centrality matrix
    BC: Nx1 np.ndarray, node betweenness centrality vector

    """
    n = len(CM)
    BC = np.zeros((n,))
    EBC = np.zeros((n, n))

    for u in range(n):
        D = np.tile(np.inf, n)
        D[u] = 0  # Distance from u to itself
        NP = np.zeros((n,))
        NP[u] = 1  # Number of shortest paths
        S = np.ones((n,), dtype=bool)  # Distance permanence
        P = np.zeros((n, n))  # Predecessors matrix
        Q = np.zeros((n,), dtype=int)
        q = n - 1

        CM_temp = CM.copy()  # Create a temporary copy of CM
        V = [u]
        while True:
            S[V] = 0  # Mark distances to V as permanent
            CM_temp[:, V] = 0  # Remove in-edges to V
            for v in V:
                Q[q] = v
                q -= 1
                W, = np.where(CM_temp[v, :])  # Find neighbors
                for w in W:
                    Duw = D[v] + CM_temp[v, w]
                    if Duw < D[w]:
                        D[w] = Duw
                        NP[w] = NP[v]
                        P[w, :] = 0
                        P[w, v] = 1
                    elif Duw == D[w]:
                        NP[w] += NP[v]
                        P[w, v] = 1

            if D[S].size == 0:
                break  # All nodes reached
            if np.isinf(np.min(D[S])):
                Q[:q + 1], = np.where(np.isinf(D))  # Mark unreachable nodes
                break
            V, = np.where(D == np.min(D[S]))

        DP = np.zeros((n,))  # Dependency values
        for w in Q[:n - 1]:
            BC[w] += DP[w]
            for v in np.where(P[w, :])[0]:
                DPvw = (1 + DP[w]) * NP[v] / NP[w]
                DP[v] += DPvw
                EBC[v, w] += DPvw

    return EBC, BC

def participation_coef(CM: np.ndarray) -> np.ndarray:
    """
    Computes the participation coefficient for nodes.

    Args:
    ---------------
    CM: NxN np.ndarray, binary/weighted directed/undirected connection matrix

    Returns:
    ---------------
    P: Nx1 np.ndarray, participation coefficient

    """
    G = nx.from_numpy_array(CM)
    partition = community_louvain.best_partition(G)
    ci = np.array([partition[node] for node in G.nodes()])
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    n = len(CM)  # Number of nodes
    Ko = np.sum(CM, axis=1)  # Node degree
    Gc = np.dot((CM != 0), np.diag(ci))  # Community affiliation
    Kc2 = np.zeros((n,))

    for i in range(1, int(np.max(ci)) + 1):
        Kc2 += np.square(np.sum(CM * (Gc == i), axis=1))

    P = np.ones((n,)) - Kc2 / np.square(Ko)
    P[np.where(Ko == 0)] = 0  # Handle nodes with no connections

    return P

def module_degree_zscore(CM: np.ndarray) -> np.ndarray:
    """
    Computes the within-module degree z-score for nodes.

    Args:
    ---------------
    CM: NxN np.ndarray, binary/weighted directed/undirected connection matrix

    Returns:
    ---------------
    Z: Nx1 np.ndarray, within-module degree Z-score

    """
    G = nx.from_numpy_array(CM)
    partition = community_louvain.best_partition(G)
    ci = np.array([partition[node] for node in G.nodes()])
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    n = len(CM)
    Z = np.zeros((n,))
    for i in range(1, int(np.max(ci) + 1)):
        Koi = np.sum(CM[np.ix_(ci == i, ci == i)], axis=1)
        Z[np.where(ci == i)] = (Koi - np.mean(Koi)) / np.std(Koi)

    Z[np.isnan(Z)] = 0  # Handle NaN values
    return Z

def eigenvector_centrality(CM: np.ndarray) -> np.ndarray:
    """
    Computes eigenvector centrality for nodes.

    Args:
    ---------------
    CM: NxN np.ndarray, connection matrix

    Returns:
    ---------------
    eig_centrality: Nx1 np.ndarray, eigenvector centrality

    """
    from scipy import linalg
    vals, vecs = linalg.eig(CM)
    i = np.argmax(vals)  # Identify the largest eigenvalue
    eig_centrality = np.abs(vecs[:, i])
    return eig_centrality