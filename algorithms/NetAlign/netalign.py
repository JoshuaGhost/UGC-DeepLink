import scipy.sparse as sps
import numpy as np
from .. import bipartitewrapper as bmw
from scipy.sparse import csr_matrix, csc_matrix

# original code https://www.cs.purdue.edu/homes/dgleich/codes/netalign/


def make_squares(A: csr_matrix, B: csr_matrix, L: csr_matrix):
    m = np.shape(A)[0]
    n = np.shape(B)[0]

    rpA = A.indptr
    ciA = A.indices
    rpB = B.indptr
    ciB = B.indices

    rpAB = L.indptr
    ciAB = L.indices
    vAB = L.data
    # print(rpAB)
    # print(ciAB)
    # print(vAB)

    Se1 = []
    Se2 = []

    wv = -np.ones(n, int)
    for i in range(m):
        # label everything in to i in B
        # label all the nodes in B that are possible matches to the current i
        possible_matches = list(range(rpAB[i], rpAB[i + 1]))
        # get the exact node ids via ciA[possible_matches]
        # wv: nodes in B -> edge pointer in L
        wv[ciAB[possible_matches]] = possible_matches
        for ri1 in range(rpA[i], rpA[i + 1]):
            # get the actual node index
            ip = ciA[ri1]
            if i == ip:
                continue
            # for node index ip, check the nodes that its related to in L
            # ri2: edge pointer in L, but the edges starts from ip. ip is the neighbor of i
            for ri2 in range(rpAB[ip], rpAB[ip + 1]):
                if ri2 == 9:
                    a = 1
                jp = ciAB[ri2]
                for ri3 in range(rpB[jp], rpB[jp + 1]):
                    j = ciB[ri3]
                    if j == jp:
                        continue
                    if wv[j] >= 0:
                        # we have a square!
                        # push!(Se1, ri2)
                        # push!(Se2, wv[j])
                        Se1.append(ri2)
                        Se2.append(wv[j])
        # remove labels for things in in adjacent to B
        wv[ciAB[possible_matches]] = -1

    Le = np.zeros((L.getnnz(), 2), int)
    LeWeights = np.zeros(L.getnnz(), float)
    for i in range(0, m):
        j = list(range(rpAB[i], rpAB[i + 1]))
        Le[j, 0] = i
        Le[j, 1] = ciAB[j]
        LeWeights[j] = vAB[j]
    Se = np.zeros((len(Se1), 2), int)
    Se[:, 0] = Se1
    Se[:, 1] = Se2
    # se1 和 se2 都是L的边的index
    return Se, Le, LeWeights


def netalign_setup(A: csr_matrix, B: csr_matrix, L: csr_matrix):  # needs fix
    Se, Le, LeWeights = make_squares(A, B, L)
    # li and lj are the rows and columns of the similarity matrix L
    li = Le[:, 0]
    lj = Le[:, 1]
    # Se1 and Se2 are the rows and columns of the squares. Se1和Se2都用了L的边的index
    # 所以S矩阵是边<->边的对应关系, 一个对应indicates这里有一个square
    Se1 = Se[:, 0]
    Se2 = Se[:, 1]
    values = np.ones(len(Se1))
    # print(values)
    el = L.getnnz()
    # S 矩阵是“square”矩阵，它的行是源图的边，列是目标图的边，如果这两条边能够组成一个square（即四个定点对应有两组match），则值是1
    # print(el)
    # print(len(Se1))
    # print(Se1)
    # print(len(Se2))
    # print(Se2)
    # print(sps.find(L))
    S = csc_matrix(
        (
            values,
            (
                Se1,
                Se2,
            ),
        ),
        (el, el),
    )
    # print(f'S:\n{S.toarray()}')
    # print(f'S.triu:\n{sps.triu(S, 1).toarray()}')

    return S, LeWeights, li, lj


def othermaxplus(dim, li, lj, lw, m, n):
    if dim == 1:
        i1 = lj
        i2 = li
        N = n
    else:
        i1 = li
        i2 = lj
        N = m

    dimmax1 = np.zeros(N)
    dimmax2 = np.zeros(N)
    dimmaxind = np.zeros(N)
    nedges = len(li)

    for i in range(nedges):
        if lw[i] > dimmax2[i1[i]]:
            if lw[i] > dimmax1[i1[i]]:
                dimmax2[i1[i]] = dimmax1[i1[i]]
                dimmax1[i1[i]] = lw[i]
                dimmaxind[i1[i]] = i2[i]
            else:
                dimmax2[i1[i]] = lw[i]

    omp = np.zeros(len(lw))
    for i in range(nedges):
        if i2[i] == dimmaxind[i1[i]]:
            omp[i] = dimmax2[i1[i]]
        else:
            omp[i] = dimmax1[i1[i]]

    return omp


def othersum(si, sj, s, m, n):
    rowsum = accumarray(si, s, m)
    return rowsum[si] - s


def accumarray(xij, xw, n):
    sums = np.zeros(n)

    for i in range(len(xij)):
        sums[xij[i]] += xw[i]

    return sums


def netalign_main(data, a=1, b=1, gamma=0.99, dtype=2, maxiter=100, verbose=True):
    """
    NetAlign algorithm
    :param data: dictionary with keys 'S', 'li', 'lj', 'w'
        key 'S': square matrix S
        key 'li': list of indices of rows of S
        key 'lj': list of indices of columns of S
        key 'w': list of weights of edges
    :param a: alpha
    :param b: beta
    :param gamma: gamma
    :param dtype: 1 for maxplus, 2 for maxsum
    :param maxiter: maximum number of iterations
    :param verbose: print progress
    :return: dictionary with keys 'x', 'y', 'objective', 'iter', 'time'
    """

    S = data["S"]
    li = data["li"]
    lj = data["lj"]
    w = data["w"]

    S = sps.csr_matrix(S)

    nedges = len(li)
    nsquares = S.count_nonzero() // 2

    # print(f'S: {S.toarray()}')
    # print(f'S.triu: {sps.triu(S, 1).toarray()}')

    # compute a vector that allows us to transpose data between squares.
    # triu(S, 1): 返回S的上三角矩阵，不包括对角线
    sui, suj, _ = sps.find(sps.triu(S, 1))
    # SI最终非零的元素行列都是S中的非零元素（对角线除外），但是每一个square都被分配了不同的index，从1开始
    SI = sps.csr_matrix((list(range(1, len(sui) + 1)), (sui, suj)), shape=S.shape)

    SI = SI + SI.transpose()
    # 返回SI中非零元素的行列index（si与sj）和值（sind）
    # print("SI:\n", SI.toarray())
    si, sj, sind = sps.find(SI)
    sind = [x - 1 for x in sind]
    # print("si:\n", si)
    # print("sind:\n", sind)
    # print((S.shape[0], nsquares))
    # 这行出问题
    SP = sps.csr_matrix(([1] * len(si), (si, sind)), shape=(S.shape[0], nsquares))
    sij, sijrs, _ = sps.find(SP)
    sind = list(range(SP.count_nonzero()))
    spair = sind[:]
    spair[::2] = sind[1::2]
    spair[1::2] = sind[::2]

    # Initialize the messages
    ma = np.zeros(nedges, int)
    mb = np.zeros(nedges, int)
    ms = np.zeros(S.count_nonzero())
    sums = np.zeros(nedges)

    damping = gamma
    curdamp = 1
    alpha = a
    beta = b

    fbest = 0
    fbestiter = 0
    if verbose:
        print(
            "{:4s}   {:>4s}   {:>7s} {:>7s} {:>7s} {:>7s}   {:>7s} {:>7s} {:>7s} {:>7s}".format(
                "best",
                "iter",
                "obj_ma",
                "wght_ma",
                "card_ma",
                "over_ma",
                "obj_mb",
                "wght_mb",
                "card_mb",
                "over_mb",
            )
        )

    setup, m, n = bmw.bipartite_setup(li, lj, w)

    for it in range(1, maxiter + 1):
        prevma = ma
        prevmb = mb
        prevms = ms
        prevsums = sums
        curdamp = damping * curdamp

        omaxb = np.array(
            [max(0, x) for x in othermaxplus(2, li, lj, mb, m, n)],
        )
        omaxa = np.array(
            [max(0, x) for x in othermaxplus(1, li, lj, ma, m, n)],
        )

        msflip = ms[spair]
        mymsflip = msflip + beta
        mymsflip = [min(beta, x) for x in mymsflip]
        mymsflip = [max(0, x) for x in mymsflip]
        # mymsflip is F in the paper

        sums = accumarray(sij, mymsflip, nedges)
        # d^(t) in the paper

        ma = alpha * w - omaxb + sums
        mb = alpha * w - omaxa + sums

        ms = alpha * w[sij] - (omaxb[sij] + omaxa[sij])
        ms += othersum(sij, sijrs, mymsflip, nedges, nsquares)

        if dtype == 1:
            ma = curdamp * (ma) + (1 - curdamp) * (prevma)
            mb = curdamp * (mb) + (1 - curdamp) * (prevmb)
            ms = curdamp * (ms) + (1 - curdamp) * (prevms)
        elif dtype == 2:
            ma = ma + (1 - curdamp) * (prevma + prevmb - alpha * w + prevsums)
            mb = mb + (1 - curdamp) * (prevmb + prevma - alpha * w + prevsums)
            ms = ms + (1 - curdamp) * (prevms + prevms[spair] - beta)
        elif dtype == 3:
            ma = curdamp * ma + (1 - curdamp) * (prevma + prevmb - alpha * w + prevsums)
            mb = curdamp * mb + (1 - curdamp) * (prevmb + prevma - alpha * w + prevsums)
            ms = curdamp * ms + (1 - curdamp) * (prevms + prevms[spair] - beta)

        hista = bmw.round_messages(ma, S, w, alpha, beta, setup, m, n)[:-2]

        histb = bmw.round_messages(mb, S, w, alpha, beta, setup, m, n)[:-2]

        if hista[0] > fbest:
            fbestiter = it
            mbest = ma
            fbest = hista[0]
        if histb[0] > fbest:
            fbestiter = -it
            mbest = mb
            fbest = histb[0]

        if verbose:
            if fbestiter == it:
                bestchar = "*a"
            elif fbestiter == -it:
                bestchar = "*b"
            else:
                bestchar = ""
            print(
                "{:4s}   {:4d}   {:7.2f} {:7.2f} {:7d} {:7d}   {:7.2f} {:7.2f} {:7d} {:7d}".format(
                    bestchar, it, *hista, *histb
                )
            )

    return sps.csr_matrix((mbest, (li, lj)))
