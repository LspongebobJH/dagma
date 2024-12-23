"""
General knockoff generation, considering the whole X (n*p) as a design matrix, and 
generating X_tilde (n*p) for the whole X.
"""

import utils, utils_dagma, pickle, os, yaml
import numpy as np
import networkx as nx
from argparse import ArgumentParser
from knockoff_diagn import _get_single_clf, _adjust_marginal
from tqdm import tqdm
from sklearn.linear_model import Lasso, ElasticNet
from copy import deepcopy

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dst_data_version', type=str, required=True)
    parser.add_argument('--dst_data_name', type=str, required=True)
    parser.add_argument('--fit_W_version', type=str)
    parser.add_argument('--fit_W_name', type=str)
    parser.add_argument('--option', type=int, required=True, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    parser.add_argument('--method_diagn_gen', type=str, required=True, choices=['lasso', 'xgb', 'elastic', 'OLS_cuda', "PLS"])
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
    rng = np.random.default_rng(seed=configs['seed_knockoff'])

    assert configs['dedup'] is True
    
    if configs['s0'] is None:
        configs['s0'] = configs['d'] * 4


    output_data_dir = os.path.join(
        data_dir,
        configs['dst_data_version'],
        configs['dst_data_name'],
        "knockoff"
    )
    output_data_path = os.path.join(output_data_dir, f'knockoff_{configs["seed_X"]}_{configs["seed_knockoff"]}.pkl')
    output_config_path = os.path.join(output_data_dir, f'knockoff_{configs["seed_X"]}_{configs["seed_knockoff"]}_configs.yaml')

    if (os.path.exists(output_data_path) or os.path.exists(output_data_path)) and not configs['force_save']:
        raise Exception(f"{output_data_dir} already exists.")

    # load X
    # version = f"v11/v{configs['src_data_version']}"
    version=f"{configs['dst_data_version']}/{configs['dst_data_name']}"
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
    if configs['option'] not in [5, 10, 16, 17] or (configs['option'] == 10 and configs['topo_sort']):
        if configs['W_type'] == 'W_est':
            version = f"{configs['fit_W_version']}/{configs['n']}_{configs['d']}_{configs['s0']}/{configs['fit_W_name']}.pkl"
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

    # fit knockoff
    X_tilde = np.zeros_like(X)
    if configs['topo_sort']:
        nodes = list(nx.topological_sort(G))
    else:
        nodes = list(range(X.shape[1]))
    
    for _idx, j in enumerate(tqdm(nodes)):
        no_need_pred = False
        no_input = False
        preds = 0.
        X_input = None
        """
        A: parents
        B: children
        C: parents of children
        A+B+C: markov blanket
        """
        if configs['option'] in [1, 13]: # A + B + C (+ knockoff) fit j
            if configs['dedup']:
                n_list = []
                for func in [parents_X, children_X, parents_of_children_X]:
                    _, _n = func(G, j, X, return_idx=True)
                    if _n is not None:
                        n_list.append(_n)
                if len(n_list) > 0:
                    no_input = False
                    n_list = list(set(np.concatenate(n_list).tolist()))
                    X_input = X[:, n_list]
                else:
                    no_input = True
            else:
                raise NotImplementedError("non-deduplicated version is not valid")
                X_input = [
                    func(G, j, X) for func in [parents_X, children_X, parents_of_children_X]
                ]
                no_input = True
                for _input in X_input:
                    if _input is not None:
                        no_input = False
                        break
            if configs['option'] == 13:
                if _idx > 0:
                    # find intersection of nodes with knockoff and A+B+C
                    n_knockoff = intersection(nodes[:_idx], n_list)
                    if len(n_knockoff) > 0:
                        X_input = np.concatenate(
                            [X_input, X_tilde[:, n_knockoff]],
                            axis=1
                        )

        elif configs['option'] == 2: # B + C
            X_input = [
                func(G, j, X) for func in [children_X, parents_of_children_X]
            ]
            no_input = True
            for _input in X_input:
                if _input is not None:
                    no_input = False
                    break

        elif configs['option'] == 3: # B - w(C -> B)*C
            child_X, child = children_X(G, j, X, True)
            par_of_child_X, par_of_child = parents_of_children_X(G, j, X, True)
            if child is None:
                no_input = True
            elif par_of_child is None and child is not None:
                X_input = child_X
            else:
                _W = W[par_of_child, :][:, child]
                X_input = child_X - par_of_child_X @ _W

        elif configs['option'] == 4: # A + B - w(C -> B)*C
            par_X, par = parents_X(G, j, X, True)
            child_X, child = children_X(G, j, X, True)
            par_of_child_X, par_of_child = parents_of_children_X(G, j, X, True)

            if child is None:
                X_input = None
            elif par_of_child is None and child is not None:
                X_input = child_X
            else:
                _W = W[par_of_child, :][:, child]
                X_input = child_X - par_of_child_X @ _W
            
            if par is not None:
                if X_input is None:
                    X_input = par_X
                else:
                    X_input = np.concatenate([X_input, par_X], axis=1)
            else:
                if X_input is None:
                    no_input = True
                else:
                    pass # X_input kept as is

        elif configs['option'] in [5, 10]: # all nodes (+ knockoff) fit j
            p = X.shape[1]
            input_idx = np.array([i for i in np.arange(0, p) if i != j])
            X_input = X[:, input_idx]
            if configs['option'] == 10:
                if _idx > 0:
                    X_input = np.concatenate(
                        [X_input, X_tilde[:, nodes[:_idx]]],
                        axis=1
                    )
        
        elif configs['option'] == 7: # only parents (only A)
            X_input = [
                func(G, j, X) for func in [parents_X]
            ]
            no_input = True
            for _input in X_input:
                if _input is not None:
                    no_input = False
                    break

        elif configs['option'] == 8: # all nodes or after dag control, but dagma knockoffdiagn rather than OLS or Lasso
            preds = X @ W
            preds = preds[:, j]
            no_need_pred = True

        elif configs['option'] == 9: # only ancestors
            X_input = [
                func(G, j, X) for func in [ancestors_X]
            ]
            no_input = True
            for _input in X_input:
                if _input is not None:
                    no_input = False
                    break

        elif configs['option'] == 11: # A + B
            X_input = [
                func(G, j, X) for func in [parents_X, children_X]
            ]
            no_input = True
            for _input in X_input:
                if _input is not None:
                    no_input = False
                    break

        elif configs['option'] == 12: # similar to 9, only ancestors + ancestors' knockoff, nodes sorted by topo sort such that ancestors' knockoff exist.
            X_input, ancestors_n = ancestors_X(G, j, X, return_idx=True)
            if X_input is None:
                no_input = True
            else:
                no_input = False
                X_input = np.concatenate(
                    [X_input, X_tilde[:, ancestors_n]],
                    axis=1
                )

        elif configs['option'] in [14, 15]: # non-descendants
            X_input, non_descendants_n = non_descendants_X(G, j, X, return_idx=True)
            if X_input is None:
                no_input = True
            else:
                no_input = False
            if configs['option'] == 15:
                if _idx > 0 and non_descendants_n is not None:
                    # find intersection of nodes with knockoff and A+B+C
                    n_knockoff = intersection(nodes[:_idx], non_descendants_n)
                    if len(n_knockoff) > 0:
                        X_input = np.concatenate(
                            [X_input, X_tilde[:, n_knockoff]],
                            axis=1
                        )
        
        elif configs['option'] == 16: # permutation
            no_need_pred = False
            no_input = True

        elif configs['option'] == 17: # gaussian noise
            sample = rng.normal(X[:, j].mean(), X[:, j].std(), size=X[:, j].shape)

        if configs['option'] != 17:
            if not no_need_pred: # need to fit model for X_pred
                if not no_input: # has no X_input for model fitting
                    if isinstance(X_input, list):
                        X_input = [val for val in X_input if val is not None]
                        X_input = np.concatenate(X_input, axis=1)
                    _configs = deepcopy(configs)
                    if _configs['method_diagn_gen'] == 'PLS' and X_input.shape[1] < configs['PLS_n_comp']:
                        _configs['method_diagn_gen'] = 'OLS_cuda'
                        print("use OLS_cuda rather than PLS")
                    preds = _get_single_clf(X_input, X[:, j], 
                                            method=_configs['method_diagn_gen'], 
                                            alpha=_configs['lasso_alpha'],
                                            n_comp=_configs['PLS_n_comp'],
                                            device=_configs['device'])
                else:
                    preds = 0.
            residuals = X[:, j] - preds
            indices_ = np.arange(residuals.shape[0])
            np.random.shuffle(indices_)
            sample = preds + residuals[indices_]
        sample = _adjust_marginal(sample, X[:, j], discrete=False)
        X_tilde[:, j] = sample
    
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)

    with open(output_data_path, 'wb') as f:
        pickle.dump(X_tilde, f)
    with open(output_config_path, 'w') as f:
        yaml.dump(configs, f)

    print("DONE!")