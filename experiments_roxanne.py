from collections import defaultdict
import json
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from sympy import csc
from torch import gt
from algorithms import bipartiteMatching
from algorithms.NetAlign.netalign import netalign_main, netalign_setup
from algorithms.isorank.isorank import isorank, isorank_greedy

from generation import similarities_preprocess

from typing import Tuple, List, Dict
import itertools 
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score


def convert_edge_list_to_csr(Ai, Aj) -> scipy.sparse.csr_matrix:
    n = max(max(Ai), max(Aj)) + 1
    nedges = len(Ai)
    Aw = np.ones(nedges)
    A = csr_matrix((Aw, (Ai, Aj)), shape=(n, n), dtype=int)
    A = A + A.T  # make undirected
    return A


def numerate(node_names, nodes_name_to_id):
    return [nodes_name_to_id[name] for name in node_names]


def load_csr_from_json_string(
    json_string,
) -> Tuple[csr_matrix, List[str], Dict[str, int]]:
    temp = eval(json_string)
    Ai, Aj = list(zip(*[[_["source"], _["target"]] for _ in temp["edges"]]))
    nodes_id_to_name = sorted(list(set(Ai).union(set(Aj))))
    nodes_name_to_id = defaultdict(int)
    for i, name in enumerate(nodes_id_to_name):
        nodes_name_to_id[name] = i
    Ai = numerate(Ai, nodes_name_to_id)
    Aj = numerate(Aj, nodes_name_to_id)
    return convert_edge_list_to_csr(Ai, Aj), nodes_id_to_name, nodes_name_to_id


def load_csr_graphs(fname):
    with open(fname, "r") as f:
        lines = f.readlines()
    A, node_id_to_name_a, node_name_to_id_a = load_csr_from_json_string(
        lines[0].strip()
    )
    # print(A.shape, A.toarray())
    # print(node_name_to_id_a, node_id_to_name_a)
    B, node_id_to_name_b, node_name_to_id_b = load_csr_from_json_string(
        lines[1].strip()
    )
    # print(B.shape, B.toarray())
    # print(node_name_to_id_b, node_id_to_name_b)

    return (
        A,
        B,
        node_id_to_name_a,
        node_name_to_id_a,
        node_id_to_name_b,
        node_name_to_id_b,
    )


def load_gts(node_name_to_id_a, node_name_to_id_b):
    with open("gts.txt", "r") as f:
        lines = f.readlines()
    gt_train = [
        [node_name_to_id_a[a], node_name_to_id_b[b]] for a, b in eval(lines[0].strip())
    ]
    gt_val = [
        [node_name_to_id_a[a], node_name_to_id_b[b]] for a, b in eval(lines[1].strip())
    ]

    # gt_val = {node_name_to_id_a[a]: node_name_to_id_b[b] for a, b in eval(lines[1].strip())}
    gt_test = {
        node_name_to_id_a[a]: node_name_to_id_b[b] for a, b in eval(lines[2].strip())
    }
    return gt_train, gt_val, gt_test


def fullfill_with_ground_truth(L, gt, value=1):
    for a, b in gt:
        L[a, b] = value
    return L


def mrr(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred)
    return np.mean(1 / (np.argwhere(y_true == y_pred)[:, 1] + 1))


def success_at_k(y_true, y_pred, k=2):
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred)
    return np.mean(np.any(y_true == y_pred[:, :k], axis=1))


def evaluate(Y_pred_cls, len_targets):
    def convert_to_ranking(pred_cls):
        return pred_cls.argsort(axis=1)[:, ::-1]

    Y_label_cls = np.eye(len_targets)

    Y_label_ranking = list(range(len_targets))
    Y_pred_ranking = convert_to_ranking(Y_pred_cls)

    result = {
        "AUC": roc_auc_score(Y_label_cls, Y_pred_cls),
        # MRR is equivalent to MAP in this case, because the ranking is binary
        "MRR": mrr(Y_label_ranking, Y_pred_ranking),
        "Success_at_k": success_at_k(Y_label_ranking, Y_pred_ranking, k=2),
    }
    return result


def netalign_test_setup(L, gt_test):
    temp = sorted([[a, b] for a, b in gt_test.items()], key=lambda x: x[1])
    prod_test = list(itertools.product(*zip(*temp)))
    L = fullfill_with_ground_truth(L, prod_test, 1./len(gt_test))
    return L

def load_prior_embeddings():
    ebd_fname = "individuals_info+word.json"
    with open(ebd_fname, 'r') as fin:
        loaded = json.load(fin)
    embeddings = {k: {kk: np.array(vv['metadata']['embedding']) for kk, vv in v.items()} for k, v in loaded.items()}
    return embeddings["roxdv3"], embeddings["roxhood"]


def tf_idf_analyzer(gt_train: List[List[str]], gt_test: Dict[str, str], 
                    node_id_to_name_a: List[str], node_id_to_name_b: List[str]):
    prior_ebd1, prior_ebd2 = load_prior_embeddings()

    print(gt_train)
    print(gt_test)
    train_labels = sorted(gt_train)
    test_labels = sorted(list(gt_test.items()))
    X_train = [prior_ebd1[node_id_to_name_a[node1]] for node1, _ in train_labels]
    Y_train = [prior_ebd2[node_id_to_name_b[node2]] for _, node2 in train_labels]
    
    model = LinearRegression().fit(X_train, Y_train)
    
    X_test = [prior_ebd1[node_id_to_name_a[node1]] for node1, _ in test_labels]
    Y_test = [prior_ebd2[node_id_to_name_b[node2]] for _, node2 in test_labels]
    Y_hat = model.predict(X_test)

    Y_pred_cls = cosine_similarity(Y_hat, Y_test)
    
    # result = evaluate(Y_pred_cls, len(Y_test))
    
    return Y_pred_cls


def get_sim_for_test(old_sim, gt_test):
    temp = sorted([[a, b] for a, b in gt_test.items()], key=lambda x: x[1])
    a_indices, b_indices = list(zip(*temp))
    return (old_sim[a_indices, :])[:, b_indices]


if __name__ == "__main__":
    import random
    random.seed(6)
    np.random.seed(6)
    (
        A,
        B,
        node_id_to_name_a,
        node_name_to_id_a,
        node_id_to_name_b,
        node_name_to_id_b,
    ) = load_csr_graphs("networks.txt")
    gt_train, gt_val, gt_test = load_gts(node_name_to_id_a, node_name_to_id_b)
    # algorithm_name = 'isorank'
    algorithm_name = 'tf_idf'
    # algorithm_name = "netalign"
    if algorithm_name == "isorank":
        L = similarities_preprocess.create_L(A, B, lalpha=2)
        # the L matrix indicates the similarity between nodes in A and B. But in our case, we have training and val data from the
        # ground truth. So we can use that to fulfill the L matrix.
        L = fullfill_with_ground_truth(L, gt_train)
        L = fullfill_with_ground_truth(L, gt_val)
        S = similarities_preprocess.create_S(A, B, L)
        print(gt_val)

        graph_1_nodes, graph_2_nodes, w = scipy.sparse.find(L)
        a = 0.2
        b = 0.8
        x, nzi, nzj, m, n = isorank(S, w, graph_1_nodes, graph_2_nodes, a, b)
        sim = scipy.sparse.csr_matrix((x, (graph_1_nodes, graph_2_nodes)), shape=(m, n))
        sim = get_sim_for_test(sim, gt_test)
    elif algorithm_name == "netalign":
        m, n = A.shape[0], B.shape[0]
        L = csr_matrix((m, n))
        L = fullfill_with_ground_truth(L, gt_train)
        L = fullfill_with_ground_truth(L, gt_val)
        L = netalign_test_setup(L, gt_test)
        S, LeWeights, li, lj = netalign_setup(A, B, L)
        
        data = {
            "S": S,
            "li": li,
            "lj": lj,
            "w": LeWeights,
        }
        sim = netalign_main(data)
        print(f'sim: {sim}')
        sim = get_sim_for_test(sim, gt_test)
    elif algorithm_name == "tf_idf":
        sim = tf_idf_analyzer(gt_train, gt_test, node_id_to_name_a, node_id_to_name_b)

    # temp = sorted([[a, b] for a, b in gt_test.items()], key=lambda x: x[1])
    # a_indices, b_indices = list(zip(*temp))
    # sim = sim[a_indices, :]
    # sim = sim[:, b_indices]
    sim = sim.toarray() if not isinstance(sim, np.ndarray) else sim
    print(evaluate(sim, len(gt_test)))

    # for a, b in gt_test.items():

    #     if a in gt_test:
    #         print(b)
    #         print(f"source = {a}({node_id_to_name_a[a]});"
    #          f"pred = {b}({node_id_to_name_b[b]});"
    #          f"gt = {gt_test[a]}({node_id_to_name_b[gt_test[a]]})")
    #     if a in gt_val:
    #         print(b)
    #         print(f"source = {a}({node_id_to_name_a[a]});"
    #          f"pred = {b}({node_id_to_name_b[b]});"
    #          f"gt = {gt_val[a]}({node_id_to_name_b[gt_val[a]]})")
    # # ret = [(a, b, gt_test[a]) for a, b in zip(ma, mb) if a in gt_test]
    # # print(ret)
