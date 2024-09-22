"""
Knockoff generation specific to Genie3 and GRNBoost2, fit p*n*(p-1) X_tilde,
where each n*(p-1) is X_tilde of a design matrix X (n*(p-1)) corresponding to 
a regression task.
"""

import utils, utils_dagma, pickle, os, yaml
import numpy as np
import networkx as nx
from argparse import ArgumentParser
from knockoff_diagn import _get_single_clf, _adjust_marginal
from tqdm import tqdm
from sklearn.linear_model import Lasso, ElasticNet
from copy import deepcopy

# utility function
def postprocess_res(res_list: list, self_n: int):
    res_list = list(set(res_list))
    if self_n in res_list:
        res_list.remove(self_n)
    return res_list

def ancestors_X(G: nx.DiGraph, i: int, X: np.ndarray, return_idx: bool = False):
    ancestors = list(nx.ancestors(G, i))
    ancestors = postprocess_res(ancestors, i)
    if len(ancestors) != 0:
        if return_idx:
            return X[:, ancestors], ancestors
        else:    
            return X[:, ancestors]
    else:
        print(f"No ancestors: {i}")
        if return_idx:
            return None, None
        else:
            return None

def non_descendants_X(G: nx.DiGraph, i: int, X: np.ndarray, return_idx: bool = False): # unfinished
    descendants = list(nx.descendants(G, i))
    all_n = list(G.nodes)
    non_descendants = list(
        set(all_n).difference(set(descendants))
    )
    non_descendants = postprocess_res(non_descendants, i)
    assert i not in non_descendants
    if len(non_descendants) != 0:
        if return_idx:
            return X[:, non_descendants], non_descendants
        else:    
            return X[:, non_descendants]
    else:
        print(f"No non_descendants: {i}")
        if return_idx:
            return None, None
        else:
            return None

def parents_X(G: nx.DiGraph, i: int, X: np.ndarray, return_idx: bool = False):
    parents = list(G.predecessors(i))
    parents = postprocess_res(parents, i)
    if len(parents) != 0:
        if return_idx:
            return X[:, parents], parents
        else:    
            return X[:, parents]
    else:
        print(f"No parents: {i}")
        if return_idx:
            return None, None
        else:
            return None

def children_X(G: nx.DiGraph, i: int, X: np.ndarray, return_idx: bool = False):
    children = list(G.successors(i))
    children = postprocess_res(children, i)
    if len(children) != 0:
        if return_idx:
            return X[:, children], children
        else:
            return X[:, children]
    else:
        print(f"No children: {i}")
        if return_idx:
            return None, None
        else:
            return None

def parents_of_children_X(G: nx.DiGraph, i: int, X: np.ndarray, return_idx: bool = False):
    children = list(G.successors(i))
    parents_of_children = []
    for c in children:
        parents_of_children.extend(list(G.predecessors(c)))
    parents_of_children = postprocess_res(parents_of_children, i)

    if len(parents_of_children) != 0:
        if return_idx:
            return X[:, parents_of_children], parents_of_children
        else:
            return X[:, parents_of_children]
    else:
        print(f"No parents of children: {i}")
        if return_idx:
            return None, None
        else:
            return None
    
def ego_graph_X(G: nx.DiGraph, i: int, X: np.ndarray, radius: int, return_idx: bool = False):
    ego_graph: nx.Graph = nx.ego_graph(G, i, radius, center=False)
    nodes = list(ego_graph.nodes())
    nodes = postprocess_res(nodes, i)
    if len(nodes) != 0:
        if return_idx:
            return X[:, nodes], nodes
        else:
            return X[:, nodes]
    else:
        print(f"No ego-graph with radius={radius}: {i}")
        if return_idx:
            return None, None
        else:
            return None

def intersection(n1_list: list, n2_list: list):
    return list(set(n1_list).intersection(set(n2_list)))

def knockoff_single_X(X: np.ndarray, G: nx.DiGraph, target_n: int, configs: dict):
    option, topo_sort = configs['option'], configs['topo_sort']
    
    if topo_sort:
        design_G = deepcopy(G)
        design_G.remove_node(target_n)
        nodes = list(nx.topological_sort(design_G))

    else:
        p = X.shape[1]
        design_n = [i for i in range(0, p) if i != target_n]
        nodes = design_n

    X = X[:, nodes]

    X_tilde = np.zeros_like(X)
    for j in tqdm(range(X.shape[1])):
        if option in [5, 10]:
            p = X.shape[1]
            input_idx = [i for i in range(0, p) if i != j]
            X_input = X[:, input_idx]
            if option == 10 and j > 0:
                X_input = np.concatenate(
                    [X_input, X_tilde[:, :j]],
                    axis=1
                )

            _configs = deepcopy(configs)
            if _configs['method_diagn_gen'] == 'PLS' and X_input.shape[1] < configs['PLS_n_comp']:
                _configs['method_diagn_gen'] = 'OLS_cuda'
                print("use OLS_cuda rather than PLS")
            preds = _get_single_clf(X_input, X[:, j], 
                                    method=_configs['method_diagn_gen'], 
                                    alpha=_configs['lasso_alpha'],
                                    n_comp=_configs['PLS_n_comp'],
                                    device=_configs['device'])
            
            residuals = X[:, j] - preds
            indices_ = np.arange(residuals.shape[0])
            np.random.shuffle(indices_)
            sample = preds + residuals[indices_]
            sample = _adjust_marginal(sample, X[:, j], discrete=False)
            X_tilde[:, j] = sample
    return X_tilde
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_version', type=str, required=True)
    parser.add_argument('--dst_version', type=str, required=True)
    parser.add_argument('--fit_W_version', type=str, required=True)
    parser.add_argument('--option', type=int, default=None, choices=[5, 10])
    parser.add_argument('--method_diagn_gen', type=str, default='OLS_cuda', choices=['lasso', 'xgb', 'elastic', 'OLS_cuda', "PLS"])
    parser.add_argument('--lasso_alpha', type=str, default='knockoff_diagn', choices=['knockoff_diagn', 'sklearn', 'OLS'])
    parser.add_argument('--PLS_n_comp', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--topo_sort', action="store_true", default=False)
    parser.add_argument('--dedup', action="store_true", default=False, help="deduplicate valid only when option=1. from option=14, deduplicate is forced.") # NOT DONE
    parser.add_argument('--W_type', type=str, default=None, choices=["W_true", "W_est"])
    parser.add_argument('--disable_dag_control', action='store_true', default=False, help="it's available only when W_type=W_est and option != 5")
    parser.add_argument('--seed_model', type=int, default=0, choices=[0], help="it's available only when option != 5")
    parser.add_argument('--seed_X', type=int, default=1)
    parser.add_argument('--seed_knockoff', type=int, default=1)
    parser.add_argument('--d', type=int, required=True)
    parser.add_argument('--s0', type=int, default=None)
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:7')
    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--force_save', action='store_true', default=False)

    # normalization to mitigate ill-condition of X
    parser.add_argument('--norm', type=str, choices=['col', 'row'])

    args = parser.parse_args()
    configs = vars(args)
    device = configs['device']
    data_dir = '/home/jiahang/dagma/src/dagma/simulated_data'
    # assert configs['seed_X'] == 1
    assert configs['seed_model'] == 0
    n_jobs = configs['n_jobs']
    utils.set_random_seed(configs['seed_knockoff'])

    assert configs['dedup'] is True
    
    if configs['s0'] is None:
        configs['s0'] = configs['d'] * 4


    output_data_dir = os.path.join(
        data_dir,
        configs['data_version'],
        configs['dst_version'],
        "knockoff"
    )
    output_data_path = os.path.join(output_data_dir, f'knockoff_{configs["seed_X"]}_{configs["seed_knockoff"]}.pkl')
    output_config_path = os.path.join(output_data_dir, f'knockoff_{configs["seed_X"]}_{configs["seed_knockoff"]}_configs.yaml')

    if (os.path.exists(output_data_path) or os.path.exists(output_data_path)) and not configs['force_save']:
        raise Exception(f"{output_data_dir} already exists.")

    # load X
    # version = f"v11/v{configs['src_data_version']}"
    version=f"{configs['data_version']}/{configs['dst_version']}"
    data_path = os.path.join(data_dir, version, 'X', f'X_{configs["seed_X"]}.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    X, W_true = data['X'], data['W_true']
    B_true = (W_true != 0)

    if configs['norm'] is not None:
        if configs['norm'] == 'col':
            X_mean = X.mean(axis=0, keepdims=True)
            X_std = X.std(axis=0, keepdims=True)
        elif configs['norm'] == 'row':
            X_mean = X.mean(axis=1, keepdims=True)
            X_std = X.std(axis=1, keepdims=True)
        X = (X - X_mean) / (X_std + 1e-8)

    # load W
    if configs['option'] != 5:
        if configs['W_type'] == 'W_est':
            # version = f"v39/{configs['d']}_{configs['s0']}/W_{configs['d']}_{configs['s0']}_{configs['seed_X']}_{configs['seed_model']}{configs['note']}.pkl"
            version = f"{configs['fit_W_version']}/{configs['d']}_{configs['s0']}/W_{configs['d']}_{configs['s0']}_{configs['seed_X']}_{configs['seed_model']}{configs['note']}.pkl"
            data_path = os.path.join(data_dir, version)
            with open(data_path, 'rb') as f:
                W_est = pickle.load(f)

            # preprocessing
            if configs['disable_dag_control']:
                G_est = nx.DiGraph(W_est)
            else:
                mask = utils.extract_dag_mask(np.abs(W_est), 0)
                W_est[~mask] = 0.
                G_est = nx.DiGraph(W_est)
                assert nx.is_directed_acyclic_graph(G_est)

            G = G_est
            W = W_est
        
        elif configs['W_type'] == 'W_true':
            G_true = nx.DiGraph(W_true)
            G = G_true
            W = W_true

    # fit knockoff
    X_tildes = []
    for j in range(X.shape[1]):
        if configs['topo_sort']:
            X_tildes.append(knockoff_single_X(X, G, j, configs))
        else:
            X_tildes.append(knockoff_single_X(X, None, j, configs))
        
    X_tilde = np.stack(X_tildes, axis=0)

    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)

    with open(output_data_path, 'wb') as f:
        pickle.dump(X_tilde, f)
    with open(output_config_path, 'w') as f:
        yaml.dump(configs, f)

    print("DONE!")
        
        