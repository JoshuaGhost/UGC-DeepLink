from . import ex
from . import matching, evaluation, similarities_preprocess
import numpy as np
import networkx as nx
import scipy.sparse as sps
import os
import copy
import time


def format_output(res):

    if isinstance(res, tuple):
        sim, cost = res
    else:
        sim = res
        cost = None

    if sim is not None:
        sim = sps.csr_matrix(sim)

    if cost is not None:
        cost = sps.csr_matrix(cost)

    return sim, cost


@ex.capture
def alg_exe(alg, data, args):
    return alg.main(data=data, **args)


@ ex.capture
def run_alg(_seed, data, Gt, i, algs, _log, _run, mtypes=None, mall=False):

    # random.seed(_seed)
    # np.random.seed(_seed)

    alg, args, mt = algs[i]

    _log.debug(f"{' ' + alg.__name__ +' ':#^35}")

    start = time.time()
    res = alg_exe(alg, data, args)
    _run.log_scalar(f"{alg.__name__}.alg", time.time()-start)

    sim, cost = format_output(res)

    try:
        _run.log_scalar(f"{alg.__name__}.sim.size", sim.size)
        _run.log_scalar(f"{alg.__name__}.sim.max", sim.max())
        _run.log_scalar(f"{alg.__name__}.sim.min", sim.min())
        _run.log_scalar(f"{alg.__name__}.sim.avg", sim.data.mean())
    except Exception:
        pass
    try:
        _run.log_scalar(f"{alg.__name__}.cost.size", cost.size)
        _run.log_scalar(f"{alg.__name__}.cost.max", cost.max())
        _run.log_scalar(f"{alg.__name__}.cost.min", cost.min())
        _run.log_scalar(f"{alg.__name__}.cost.avg", cost.data.mean())
    except Exception:
        pass

    if mall:
        res2 = []
        for mt in mtypes:
            try:

                start = time.time()
                ma, mb = matching.getmatching(sim, cost, mt)
                _run.log_scalar(f"{alg.__name__}.matching({mt})",
                                time.time()-start)

                res1 = evaluation.evall(ma, mb, data['Src'],
                                        data['Tar'], Gt, alg=alg.__name__)
            except Exception:
                _log.exception("")
                res1 = np.array([-1, -1, -1, -1, -1])
            res2.append(res1)
        result = np.array(res2)[:, 0]
    else:
        try:

            start = time.time()
            ma, mb = matching.getmatching(sim, cost, mt)
            _run.log_scalar(f"{alg.__name__}.matching({mt})",
                            time.time()-start)

            result = evaluation.evall(ma, mb, data['Src'],
                                      data['Tar'], Gt, alg=alg.__name__)
        except Exception:
            _log.exception("")
            result = np.array([-1, -1, -1, -1, -1])

    with np.printoptions(suppress=True, precision=4):
        _log.debug("\n%s", result.astype(float))

    _log.debug(f"{'#':#^35}")

    return result


@ ex.capture
def preprocess(Src, Tar, REGAL_args, CONE_args):
    # L = similarities_preprocess.create_L(Tar, Src)
    L = similarities_preprocess.create_L(Src, Tar)
    # L, _ = regal.main({"Src": Src, "Tar": Tar}, **REGAL_args)
    # L, _ = conealign.main({"Src": Src, "Tar": Tar}, **CONE_args)
    # S = similarities_preprocess.create_S(Tar, Src, L)
    S = similarities_preprocess.create_S(Src, Tar, L)
    li, lj, w = sps.find(L)

    return L, S, li, lj, w


@ ex.capture
def run_algs(Src, Tar, Gt, run, _seed, _run, prep=False, circular=False):

    if prep:
        start = time.time()
        L, S, li, lj, w = preprocess(Src, Tar)
        _run.log_scalar("graph.prep", time.time()-start)
    else:
        L = S = sps.eye(1)
        li = lj = w = np.empty(1)

    data = {
        'Src': Src,
        'Tar': Tar,
        'L': L,
        'S': S,
        'li': li,
        'lj': lj,
        'w': w
    }

    return np.array([run_alg(_seed, copy.deepcopy(data), Gt, i) for i in run])


def e_to_G(e, n):
    # n = np.amax(e) + 1
    nedges = e.shape[0]
    G = sps.csr_matrix((np.ones(nedges), e.T), shape=(n, n), dtype=int)
    G += G.T
    G.data = G.data.clip(0, 1)
    return G


@ ex.capture
def run_exp(G, output_path, noises, _log, _run):
    res2 = res3 = res4 = None
    res5 = []
    try:
        os.mkdir(f'{output_path}/graphs')

        for graph_number, g_n in enumerate(G):

            _log.info("Graph:(%s/%s)", graph_number + 1, len(G))

            res4 = []
            for noise_level, g_it in enumerate(g_n):

                _log.info("Noise_level:(%s/%s)", noise_level + 1, len(g_n))

                res3 = []
                for i, g in enumerate(g_it):
                    _log.info("iteration:(%s/%s)", i+1, len(g_it))

                    Src_e, Tar_e, Gt = g
                    n = Gt[0].size

                    prefix = f"{output_path}/graphs/{graph_number+1:0>2d}_{noise_level+1:0>2d}_{i+1:0>2d}"
                    Gt_m = np.c_[np.arange(n), Gt[0]]
                    np.savetxt(f"{prefix}_Src.txt", Src_e, fmt='%d')
                    np.savetxt(f"{prefix}_Tar.txt", Tar_e, fmt='%d')
                    np.savetxt(f"{prefix}_Gt.txt", Gt_m, fmt='%d')

                    src = nx.Graph(Src_e.tolist())
                    src_cc = len(max(nx.connected_components(src), key=len))
                    src_disc = src_cc < n

                    tar = nx.Graph(Tar_e.tolist())
                    tar_cc = len(max(nx.connected_components(tar), key=len))
                    tar_disc = tar_cc < n

                    if (src_disc):
                        _log.warning("Disc. Source: %s < %s", src_cc, n)
                    _run.log_scalar("graph.Source.disc", src_disc)
                    _run.log_scalar("graph.Source.n", n)
                    _run.log_scalar("graph.Source.e", Src_e.shape[0])

                    if (tar_disc):
                        _log.warning("Disc. Target: %s < %s", tar_cc, n)
                    _run.log_scalar("graph.Target.disc", tar_disc)
                    _run.log_scalar("graph.Target.n", n)
                    _run.log_scalar("graph.Target.e", Tar_e.shape[0])

                    res2 = run_algs(e_to_G(Src_e, n), e_to_G(Tar_e, n), Gt)

                    with np.printoptions(suppress=True, precision=4):
                        _log.info("\n%s", res2.astype(float))

                    res3.append(res2)

                res3 = np.array(res3)
                with np.printoptions(suppress=True, precision=4):
                    _log.debug("\n%s", res3.astype(float))
                res4.append(res3)

            res4 = np.array(res4)
            res5.append(res4)
    except:
        np.save(f"{output_path}/_res2", np.array(res2))
        np.save(f"{output_path}/_res3", np.array(res3))
        np.save(f"{output_path}/_res4", np.array(res4))
        _log.exception("")

    return np.array(res5)  # (g,n,i,alg,acc)