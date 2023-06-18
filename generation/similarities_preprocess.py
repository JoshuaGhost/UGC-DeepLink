from . import ex
from math import log2, floor
import numpy as np
import scipy.sparse as sps
import networkx as nx


def e_to_G(e):
    n = np.amax(e) + 1
    nedges = e.shape[0]
    G = sps.csr_matrix((np.ones(nedges), e.T), shape=(n, n), dtype=int)
    G += G.T
    G.data = G.data.clip(0, 1)
    return G


def test1():
    A = sps.csr_matrix(
        [
            [0, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0]
        ]
    )

    B = sps.csr_matrix(
        [
            [0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0]
        ]
    )

    L = sps.csr_matrix(
        [
            [0.6, 0.9, 0.3, 0.1, 0.0],
            [0.9, 0.6, 0.0, 0.0, 0.0],
            [0.3, 0.0, 0.5, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.4, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, 0.0, 0.0]
        ]
    )

    S = sps.csr_matrix(
        [
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        ]
    )

    assert np.array_equal(S.A, create_S(A, B, L).A)

    return A, B, L, S


def create_S(A:sps.csr_matrix, B:sps.csr_matrix, L:sps.csr_matrix) -> sps.csr_matrix:
    """
    CSR: Compressed Sparse Row,
        where indptr: row indices, indices: column indices, data: values
    In the paper of Global alignment of multiple protein interaction networks with application to functional orthology detection (2013)
    S is defined as:
    
    S = [Sij] = [1 if (i, j) in E(A) and (i', j') in E(B) and L(i, i') > 0 else 0]
    """
    # return None
    n = A.shape[0]
    m = B.shape[0]

    # the column indices for row i are stored in indices[indptr[i]:indptr[i+1]]
    csr_pointers, nodes_in_B = L.indptr, L.indices
    # nodes_in_A == nodes_in_B since each element from one group together indicates a correspondance between nodes from A and B
    nedges = len(nodes_in_B)

    Si = []
    Sj = []
    # a vector in the length of m (number of nodes in B), initialized by -1
    wv = np.full(m, -1)
    ri1 = 0
    for i in range(n):
        # ranges through all nodes in A
        # print(f'{i}/{n}')
        for ri1 in range(csr_pointers[i], csr_pointers[i+1]):
            # rages through pointers of all neighbors of node i in L
            # nodes_in_B[ri1] indicates the corresponding node in B
            wv[nodes_in_B[ri1]] = ri1

        for ip in A[i].nonzero()[1]:
            # in graph A, for each neighbor of node i (except i itself)
            if i == ip:
                continue
            # for jp in L[ip].nonzero()[1]:
            # print(ip)
            for ri2 in range(csr_pointers[ip], csr_pointers[ip+1]):
                jp = nodes_in_B[ri2]
                for j in B[jp].nonzero()[1]:
                    if j == jp:
                        continue
                    if wv[j] >= 0:
                        Si.append(ri2)
                        Sj.append(wv[j])
        for ri1 in range(csr_pointers[i], csr_pointers[i+1]):
            wv[nodes_in_B[ri1]] = -1

    return sps.csr_matrix(([1]*len(Si), (Sj, Si)), shape=(nedges, nedges), dtype=int)
    # return sps.csr_matrix(([1]*len(Si), (Si, Sj)), shape=(nedges, nedges), dtype=int)
    # return sps.csr_matrix(np.ones((nedges, nedges)), dtype=int)


@ ex.capture
def create_L(A: np.ndarray, B, lalpha=1, min_similarity=None) -> sps.csr_matrix:
    """
    In the paper of NetAlign, the matrix L indicates the bipartite graph between the nodes in the graph A and the nodes in B.
    According to the paper, L is a matrix of size n x m, where n is the number of nodes in A and m is the number of nodes in B.
    Each row of L is a probability distribution over the nodes in B.
    The probability distribution is defined as follows:
    For each node i in A, we sort the nodes in B according to their degrees.
    We then take such nodes from B, whose degrees are mostly similar to their correspondance in A and denote they is an one-on-many
     match. The band with of tolerance is defined using floor(lalpha * log2(m)).
    The rest of the nodes in B are assigned a probability of 0.

    :param A: The adjacency matrix of the first graph.
    :param B: The adjacency matrix of the second graph.
    :param lalpha: The parameter lalpha.
    :param mind: The minimum degree of a node in A.
    :return: The matrix L.

    This matrix L corresponds to the matrix R in the IsoRank paper. (wrong!)
    """
    n = A.shape[0]
    m = B.shape[0]

    if lalpha is None:
        return sps.csr_matrix(np.ones((n, m)))

    a = A.sum(1)
    b = B.sum(1)
    # print(a)
    # print(b)

    # a_p = [(i, m[0,0]) for i, m in enumerate(a)]
    node_idx_and_degrees_a = list(enumerate(a))
    node_idx_and_degrees_a.sort(key=lambda x: x[1])

    # b_p = [(i, m[0,0]) for i, m in enumerate(b)]
    node_idx_and_degrees_b = list(enumerate(b))
    node_idx_and_degrees_b.sort(key=lambda x: x[1])

    ab_correspondance = [0] * n
    start = 0
    end = floor(lalpha * log2(m))
    for node_idx_a, node_degree_a in node_idx_and_degrees_a:
        # node_idx_and_degrees_a[:, 1] = 1 2 3 4 5 6 7
        # ap[1]                                ^
        #                                s   e  
        # in this case,  abs(node_idx_and_degrees_b[end][1] - ap[1]) <= abs(node_idx_and_degrees_b[start][1] - ap[1])
        #
        # node_idx_and_degrees_a[:, 1] = 1 2 3 4 5 6 7
        # ap[1]                                ^
        #                                      s   e
        # the while loop will stop at this point and the ab_m[ap[0]] will be assigned to b's nodes corresponding to [5, 6, 7]
        while(end < m and
              abs(node_idx_and_degrees_b[end][1] - node_degree_a) <= abs(node_idx_and_degrees_b[start][1] - node_degree_a)
              ):
            end += 1
            start += 1
        ab_correspondance[node_idx_a] = [bp[0] for bp in node_idx_and_degrees_b[start:end]]

    # print(ab_m)

    graph_1_nodes = []
    graph_2_nodes = []
    similarity = []
    for i, bj in enumerate(ab_correspondance):
        for j in bj:
            sim_score = 0.001
            # d = 1 - abs(a[i]-b[j]) / max(a[i], b[j])
            if min_similarity is None:
                if sim_score > 0:
                    graph_1_nodes.append(i)
                    graph_2_nodes.append(j)
                    similarity.append(sim_score)
            else:
                graph_1_nodes.append(i)
                graph_2_nodes.append(j)
                similarity.append(min_similarity if sim_score <= 0 else sim_score)
                # lw.append(0.0 if d <= 0 else d)
                # lw.append(d)

                # print(len(li))
                # print(len(lj))
                # print(len(lj))

    return sps.csr_matrix((similarity, (graph_1_nodes, graph_2_nodes)), shape=(n, m))

    # L = sps.csr_matrix((lw, (li, lj)), shape=(n, m))
    # colsums = np.sum(L.A, axis=1)
    # pi, pj, pv = sps.find(L)
    # pv = np.divide(pv, colsums[pi])
    # return sps.csc_matrix((pv, (pi, pj)), shape=(n, m))


def test2():
    for i in range(2, 8):
        print(f"### {i} ###")

        n, m = np.random.randint(2**i, 2**(i+1), size=(1, 2))[0]
        A = sps.csr_matrix(np.random.randint(2, size=(n, n)), dtype=int)
        B = sps.csr_matrix(np.random.randint(2, size=(m, m)), dtype=int)
        L = create_L(A, B)
        S = create_S(A, B, L)

        print(A.A)
        print(B.A)
        print(L.A)
        print(S.A)
        # print(n, m)
        print(A.shape)
        print(B.shape)
        print(L.shape)
        print(S.shape)


def test3():
    _lim = 6

    Src_e = np.loadtxt("data/arenas_orig.txt", int)
    # perm = np.random.permutation(np.amax(Src_e)+1)
    # print(perm)
    # print(perm[Src_e])
    # Src_e = perm[Src_e]
    # print(Src_e)
    # print(np.random.permutation(np.amax(Src_e)+1)[Src_e])

    # Src_e = np.random.permutation(np.amax(Src_e)+1)[Src_e]
    # print(Src_e)

    # print(np.amax(Src_e))
    # Src_e = np.arange(np.amax(Src_e)+1)[Src_e]

    Src_e = Src_e[np.where(Src_e < _lim, True, False).all(axis=1)]
    # print(Src_e)
    # print(np.amax(Src_e))
    # print(Src_e.shape)
    Gt = np.random.permutation(_lim)
    Tar_e = Gt[Src_e]

    Tar = e_to_G(Tar_e)
    Src = e_to_G(Src_e)

    print(Tar.A)
    print(Tar.sum(1))
    print(Src.A)
    print(Src.sum(1))
    # print(create_L(Src, Tar, 1).A)
    print(create_L(Tar, Src, 1, None).A)


if __name__ == "__main__":
    A, B, L, S = test1()
    # # # test2()

    Src = nx.gnp_random_graph(10, 0.5)
    Tar = nx.gnp_random_graph(10, 0.5)

    Src = e_to_G(np.array(Src.edges))
    Tar = e_to_G(np.array(Tar.edges))

    L = create_L(Src, Tar)
    S = create_S(Src, Tar, L)

    # print(Src.A.sum(axis=1))
    print(Src.A)
    # print(Tar.A.sum(axis=1))
    print(Tar.A)
    with np.printoptions(precision=3, suppress=True):
        # print(Src.A)
        # print(Tar.A)
        print(L.A)
        print(S.A)

    # _lim = 5
    # Src_e = np.loadtxt("data/arenas_orig.txt", int)
    # Src_e = Src_e[np.where(Src_e < _lim, True, False).all(axis=1)]
    # Gt = np.random.permutation(_lim)
    # Tar_e = Gt[Src_e]

    # Tar = e_to_G(Tar_e)
    # Src = e_to_G(Src_e)

    # # L = create_L(Src, Tar)
    # # S = create_S(Src, Tar, L)

    # print(create_S(Src, Tar, create_L(Src, Tar)).A)
    # print(create_S(Tar, Src, create_L(Tar, Src)).A)

    # # print(A.A)
    # # print(B.A)
    # # print(create_L(A, B, 1).A)
    # # print(create_L(B, A, 1).A)
