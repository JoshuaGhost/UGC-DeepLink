import numpy as np
import scipy.sparse
import scipy

from algorithms import bipartiteMatching
from algorithms.bipartiteMatching import bipartite_matching_primal_dual, bipartite_matching_setup, bipartite_matching, \
    edge_list
import math

# from data import ReadFile, similarities_preprocess
# from data.similarities_preprocess import create_L
from evaluation import evaluation
from generation import similarities_preprocess


def edge_indicator(match, ei, ej):
    ind = np.zeros(len(ei), int)
    for i in range(0, len(ei)):
        if match[ei[i]] == ej[i]:
            ind[i] = 1
    return ind


def isorank(S, w, li, lj, a=0.5, b=1, alpha=2/3, rtype=2, tol=1e-12, maxiter=100, verbose=True):
    '''
    question: can i modify the isorank to predict a rank of candidate correspondance for each node?
    '''
    nzi = li.copy()
    nzi += 1
    nzi = np.insert(nzi, [0], [0])

    nzj = lj.copy()
    nzj += 1
    nzj = np.insert(nzj, [0], [0])

    ww = np.insert(w, [0], [0])

    m = max(li) + 1
    n = max(lj) + 1

    alpha = alpha if alpha else b/(a+b)

    # P is the normalized matrix S
    P = normout_rowstochastic(S)
    csum = math.fsum(w)
    v = w/csum
    nedges = np.shape(P)[0]
    #allstats = not li and not lj
    allstats = True
    if allstats:
        rhistsize = 6
        #rp, ci, ai, tripi, m, n = bipartite_matching_setup(w,li,lj,np.amax(li),np.amax(lj))
        row_pointers, column_indices, edge_values, tripi, _, _ = bipartite_matching_setup(
            None, nzi, nzj, ww, m, n)
        # print(ci)
        # print(ai)
        # print(tripi)
        # print(m)
        # print(n)
        # matm = m
        # matn = n
        # mperm = tripi[tripi > 0]  # a permutation for the matching problem
        mperm1 = [x-1 for x in tripi if x > 0]
        mperm2 = [i for i, x in enumerate(tripi) if x > 0]
    else:
        rhistsize = 1
    r = alpha
    x = np.zeros(nedges, float) + v
    delta = 2
    it = 0
    reshist = np.zeros((maxiter+1, rhistsize), float)
    xbest = x
    fbest = 0
    fbestiter = 0
    if verbose and allstats:  # print the header
        print("{:5s}   {:4s}   {:8s}   {:7s} {:7s} {:7s} {:7}".format("best", "it",
                                                                      "pr-delta", "obj", "weight", "card", "overlap"))
    elif verbose:
        print("{:4s}   {:8s}", "iter", "delta")
    while it < maxiter and delta > tol:
        y = r * (P.T * x)
        omega = math.fsum(x) - math.fsum(y)
        y = y + omega * v
        delta = np.linalg.norm(x-y, 1)  # findd the correct one
        reshist[it] = delta
        it = it + 1
        x = y * (1/math.fsum(y))
        if allstats:
            if rtype == 1:
                xf = x
            elif rtype == 2:
                xf = a*v + b/2*(S*x)  # add v to preserve scale
            # ai = np.zeros(len(tripi), float)  # check the dimensions
            # ai[tripi > 0] = xf[mperm]
            edge_values = np.zeros(len(tripi))
            edge_values[mperm2] = xf[mperm1]
            edge_values = np.roll(edge_values, 1)
            _, _, _, noute1, match1 = bipartite_matching_primal_dual(
                row_pointers, column_indices, edge_values, tripi, m+1, n+1)
            ret_nodes_in_A = noute1-1
            # mi = bipartiteMatching.matching_indicator(
            #     rp, ci, match1, tripi, m, n)
            match1 = match1-1
            mi_int = edge_indicator(match1, li, lj)  # implement this
            # mi_int = mi[1:]
            val = np.dot(w, mi_int)
            overlap = np.dot(mi_int, (S*mi_int)/2)
            f = a*val + b*overlap
            # print(mi_int)
            # print(mi)
            if f > fbest:
                xbest = x
                fbest = f
                fbestiter = it
                itermark = "*"
            else:
                itermark = " "
            if verbose and allstats:
                print("{:5s}   {:4d}   {:8.1e}   {:5.2f} {:7.2f} {:7d} {:7d}".format(
                      itermark, it, delta, f, val, int(ret_nodes_in_A), int(overlap)))
                reshist[it, 1:-1] = [a*val + b*overlap, val, ret_nodes_in_A, overlap]
            elif verbose:
                print("{:4d}    {:8.1e}".format(it, delta))
    flag = delta > tol
    reshist = reshist[0:it, :]
    if allstats:
        x = xbest

    # print(x, flag, reshist)
    # return x, flag, reshist

    return x, nzi, nzj, m, n

def isorank_greedy(S, w, li, lj, a=0.5, b=1, alpha=2/3, rtype=2, tol=1e-12, maxiter=100, verbose=True):
    x, nzi, nzj, m, n= isorank(S, w, li, lj, a, b, alpha, rtype, tol, maxiter, verbose)

    sim = scipy.sparse.csr_matrix((x, (li, lj)), shape=(m, n))
    print(sim.toarray())

    xx = np.insert(x, [0], [0])
    m, n, val, noute, match1 = bipartiteMatching.bipartite_matching(
        None, nzi, nzj, xx)
    ret_nodes_in_A, ret_matched_nodes_in_B = bipartiteMatching.edge_list(m, n, val, noute, match1)

    return ret_nodes_in_A, ret_matched_nodes_in_B


def normout_rowstochastic(S):  # to check
    """
    This function normalizes the rows of a matrix to sum to 1
    """

    n = np.shape(S)[0]
    m = np.shape(S)[1]
    colsums = S.sum(1)
    # rwcol = np.nonzero(S)
    # pi = rwcol[0]
    # pj = rwcol[1]
    # vi = np.zeros(np.shape(pi))
    pi, pj, pv = scipy.sparse.find(S)
    # print(pi.shape)
    # for i in range(np.shape(pi)[0]):
    #     print(i)
    #     pv = S[pi, pj]
    D = colsums[pi].T
    x1 = np.true_divide(pv, D)
    x = np.ravel(x1)
    Q = scipy.sparse.csr_matrix((x, (pi, pj)), shape=(m, n))
    #n = np.shape(S)[0]
    #m = np.shape(S)[1]
    #colsums = S.sum(1)
    #magkas= S.nonzero()
    # pi=S.indices
    # pj=S.indptr
    # pv=S.data
    #pi, pj, pv = findnz(S)
    # print(m)
    # print(n)
    # print(Q.argmax(1))
    # return
    #Q= scipy.sparse.csc_matrix((pv / colsums[pi], (pi, pj)),shape = (m, n))

    # print(Q.sum(1))

    return Q



if __name__ == "__main__":
    data1 = "../../data/arenas_orig.txt"
    data2 = "../../data/noise_level_1/edges_1.txt"
    gt = "../../data/noise_level_1/gt_1.txt"

    gma, gmb = ReadFile.gt1(gt)
    # G1 = ReadFile.edgelist_to_adjmatrix1(data1)
    # G2 = ReadFile.edgelist_to_adjmatrix1(data2)
    # adj = ReadFile.edgelist_to_adjmatrixR(data1, data2)
    
    Ai, Aj = np.loadtxt(data1, int).T
    n = max(max(Ai), max(Aj)) + 1
    nedges = len(Ai)
    Aw = np.ones(nedges)
    A = scipy.sparse.csr_matrix((Aw, (Ai, Aj)), shape=(n, n), dtype=int)
    A = A + A.T # make undirected

    Bi, Bj = np.loadtxt(data2, int).T
    m = max(max(Bi), max(Bj)) + 1
    medges = len(Bi)
    Bw = np.ones(medges)
    B = scipy.sparse.csr_matrix((Bw, (Bi, Bj)), shape=(m, m), dtype=int)
    B = B + B.T

    L = similarities_preprocess.create_L(A, B, lalpha=2)
    S = similarities_preprocess.create_S(A, B, L)
    print(S)
    graph_1_nodes, graph_2_nodes, w = scipy.sparse.find(L)
    a = 0.2
    b = 0.8
    x, flag, reshist = isorank_greedy(S, w, a, b, graph_1_nodes, graph_2_nodes, 0)
    print(x, flag, reshist)
    m, n, val, noute, match1 = (
        bipartiteMatching.bipartite_matching(None, graph_1_nodes, graph_2_nodes, x))
    ma, mb = bipartiteMatching.edge_list(m, n, val, noute, match1)
    acc = evaluation.accuracy(gma+1, gmb+1, mb, ma)
    print(acc)
