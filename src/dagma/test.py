import utils, utils_dagma, pickle, os, yaml
import numpy as np
import networkx as nx
from argparse import ArgumentParser
from dagma_torch import DagmaLinear, DagmaTorch
from knockoff_diagn import get_knockoffs_stats, _get_single_clf_ko, conditional_sequential_gen_ko, _get_single_clf, _adjust_marginal
import knockoff
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.linear_model import Lasso, ElasticNet

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--exp_group_idx', type=str, default=None, choices=['v40', 'v42', 'v43', 'v44', 'v45'])
    parser.add_argument('--v40_exp_idx', type=int, default=None, choices=[0])
    parser.add_argument('--v42_i_idx', type=int, default=None, choices=[1, 2])
    parser.add_argument('--v42_ii_idx', type=int, default=None, choices=[1, 2, 3])
    parser.add_argument('--v43_method', type=str, default='elastic', choices=['lasso', 'elastic'])
    parser.add_argument('--v44_option', type=int, default=None, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--v44_option_3_radius', type=int, default=None, help="available only when v44_option=3")
    parser.add_argument('--v45_sigma', type=float, default=None)
    parser.add_argument('--v45_disable_knockoff_fit', action='store_true', default=False)
    parser.add_argument('--method_diagn_gen', type=str, default='lasso', choices=['lasso', 'xgb', 'elastic'])
    parser.add_argument('--lasso_alpha', type=str, default='knockoff_diagn', choices=['knockoff_diagn', 'sklearn', 'OLS'])
    parser.add_argument('--W_type', type=str, default=None, choices=["W_true", "W_est"])
    parser.add_argument('--disable_dag_control', action='store_true', default=False, help="it's available only when W_type=W_est")
    parser.add_argument('--seed_W', type=int, default=1, choices=[1, 2])
    parser.add_argument('--seed_X', type=int, default=1)
    parser.add_argument('--d', type=int, required=True)
    parser.add_argument('--s0', type=int, default=None)
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:7')
    parser.add_argument('--notes', type=str, default=None)

    args = parser.parse_args()
    configs = vars(args)
    device = configs['device']
    data_dir = '/home/jiahang/dagma/src/dagma/simulated_data'
    assert configs['seed_X'] == 1
    n_jobs = configs['n_jobs']
    utils.set_random_seed(configs['seed_X'])
    
    if configs['s0'] is None:
        configs['s0'] = configs['d'] * 4

    if configs['exp_group_idx'] == 'v40':

        target_dir = os.path.join(data_dir, f'v40/{configs["v40_exp_idx"]}')
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        if configs['v40_exp_idx'] == 0:
            version = f"v11/v{configs['d']}"
            data_path = os.path.join(data_dir, version, 'X', 'X_1.pkl')
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            X, W_true = data['X'], data['W_true']
            B_true = (W_true != 0)

            X_conv = X @ W_true

            eq_model = DagmaLinear(d=configs['d'], dagma_type=None, device=device, original=True).to(device)
            model = DagmaTorch(eq_model, device=device, verbose=True)
            print("fit dagma")
            W_est_no_filter, W_est = model.fit(X_conv, lambda1=0.02, lambda2=0.005, return_no_filter=True)
            acc = utils_dagma.count_accuracy(B_true, W_est != 0, use_logger=False)
            print(acc)

            data_path = os.path.join(target_dir, f"W_conv_{configs['d']}_{configs['s0']}_{configs['seed_X']}.pkl")
            with open(data_path, 'wb') as f:
                pickle.dump(W_est_no_filter, f)
            print("DONE")

    elif configs['exp_group_idx'] == 'v42':
        # load X
        version = f"v11/v{configs['d']}"
        data_path = os.path.join(data_dir, version, 'X', 'X_1.pkl')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        X, W_true = data['X'], data['W_true']
        B_true = (W_true != 0)

        # load W
        version = f"v39/W_{configs['d']}_{configs['s0']}_{configs['seed_W']}.pkl"
        data_path = os.path.join(data_dir, version)
        with open(data_path, 'rb') as f:
            W_est = pickle.load(f)

        # preprocessing
        mask = utils.extract_dag_mask(np.abs(W_est), 0)
        W_est[~mask] = 0.
        G = nx.DiGraph(W_est)
        assert nx.is_directed_acyclic_graph(G)
        
        # fit knockoff of starting nodes
        
        in_deg = G.in_degree(weight=None)
        start_n = [i for i in range(G.number_of_nodes()) if in_deg[i] == 0]
        node_set = set(G.nodes())
        non_start_n = list(node_set.difference(start_n))
        # topo_sort = nx.topological_sort(G)
        X_tilde = np.zeros_like(X)

        if configs['v42_i_idx'] == 1:
            X_start = X[:, start_n]
            p = len(start_n)
            if p < 2: # besides the target feature itself, if remained features less than 1 (that is, 0), we have no covariates to predict X_pred, thus set to 0.
                preds = np.zeros_like(X_start)
            else:
                preds = np.array(Parallel(n_jobs=n_jobs)(delayed(
                    _get_single_clf_ko)(
                        X_start, j, configs['method_diagn_gen'], configs['lasso_alpha'], configs['device']
                    ) for j in tqdm(range(p))))
                preds = preds.T
            X_tildes_start = conditional_sequential_gen_ko(X_start, preds, n_jobs=n_jobs, discrete=False, adjust_marg=True)
            X_tilde[:, start_n] = X_tildes_start

        elif configs['v42_i_idx'] == 2:
            
            def _get_knockoff_for_each_n(G: nx.DiGraph, i: int, X: np.ndarray):
                node_set = set(G.nodes())
                i_desc_set = nx.descendants(G, i)
                remained_n = node_set.difference(i_desc_set)
                remained_n.remove(i)
                remained_n = list(remained_n)
                # when all nodes are descendants of i, no covariates can be used to regress node i
                if len(remained_n) < 1: 
                    preds = np.zeros_like(X[:, i])
                else:
                    X_remained = X[:, remained_n]
                    preds = _get_single_clf(X_remained, X[:, i])

                residuals = X[:, i] - preds
                indices_ = np.arange(residuals.shape[0])
                np.random.shuffle(indices_)
                sample = preds + residuals[indices_]
                sample = _adjust_marginal(sample, X[:, i], discrete=False)
                return sample
            
            X_tildes_start = np.array(Parallel(n_jobs=n_jobs)(delayed(
                _get_knockoff_for_each_n)(G, i, X) for i in tqdm(start_n)
            )).T
            X_tilde[:, start_n] = X_tildes_start


        # fit knockoff of other nodes
        if configs['v42_ii_idx'] == 1:
            with open('./configs_dagma.yaml', 'r') as f:
                _configs = yaml.safe_load(f)
            _configs = utils.combine_configs(_configs, configs)
            _configs['method_diagn_gen'] = 'dagma'
            _configs['d'] = len(non_start_n)
            X_tilde_non_start = get_knockoffs_stats(X[:, non_start_n], _configs)
            X_tilde[:, non_start_n] = X_tilde_non_start

        if configs['notes'] is None:
            _middle_dir = f'v{configs["v42_i_idx"]}_{configs["v42_ii_idx"]}'
        else:
            _middle_dir = f'v{configs["v42_i_idx"]}_{configs["v42_ii_idx"]}_{configs["notes"]}'
        data_dir = os.path.join(data_dir, f'v42/{_middle_dir}/v{configs["d"]}_{configs["s0"]}_{configs["seed_W"]}/knockoff')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        data_path = os.path.join(data_dir, f"knockoff_1.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump(X_tilde, f)

        config_path = os.path.join(data_dir, f"knockoff_1_configs.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(configs, f)

    elif configs['exp_group_idx'] == 'v43':
        # load X
        version = f"v11/v{configs['d']}"
        data_path = os.path.join(data_dir, version, 'X', 'X_1.pkl')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        X, W_true = data['X'], data['W_true']
        B_true = (W_true != 0)

        # load W
        version = f"v39/W_{configs['d']}_{configs['s0']}_{configs['seed_W']}.pkl"
        data_path = os.path.join(data_dir, version)
        with open(data_path, 'rb') as f:
            W_est = pickle.load(f)

        # preprocessing
        if configs['disable_dag_control']:
            G = nx.DiGraph(W_est) 
        else:
            mask = utils.extract_dag_mask(np.abs(W_est), 0)
            W_est[~mask] = 0.
            G = nx.DiGraph(W_est)
            assert nx.is_directed_acyclic_graph(G)

        # fit pull knockoff
        nodes = list(G.nodes())
        X_tilde = np.zeros_like(X)
        for j in tqdm(nodes):
            _W_est = W_est.copy()
            _X = X.copy()
            idx = np.array([i for i in np.arange(0, configs['d']) if i != j])
            desc = list(nx.descendants_at_distance(G, j, 1))
            # ending nodes, for now we just assume no covariates can regress them
            # to create knockoff,
            # so simply let them be their own permutation. I also expect this operation
            # will make high-degree nodes have Z distribution.
            if len(desc) == 0: 
                indices_ = np.arange(X.shape[0])
                np.random.shuffle(indices_)
                X_tilde[:, j] = X[indices_][:, j]
            else:
                _W_est = _W_est[idx][:, desc]
                _X = _X[:, idx]
                Y1 = X[:, desc]
                Y2 = _X @ _W_est
                Y = Y1 + Y2
                if configs['v43_method'] == 'lasso':
                    clf = Lasso()
                else:
                    clf = ElasticNet()
                clf.fit(W_est[j, desc].reshape(-1, 1), Y.T)
                X_tilde[:, j] = clf.coef_.flatten()

        if configs["disable_dag_control"]:
            data_dir = os.path.join(data_dir, f'v43/v{configs["d"]}_{configs["s0"]}_{configs["seed_W"]}_{configs["v43_method"]}_disable_dag_control/knockoff')    
        else:
            data_dir = os.path.join(data_dir, f'v43/v{configs["d"]}_{configs["s0"]}_{configs["seed_W"]}_{configs["v43_method"]}/knockoff')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        data_path = os.path.join(data_dir, f"knockoff_1.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump(X_tilde, f)

        config_path = os.path.join(data_dir, f"knockoff_1_configs.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(configs, f)

    elif configs['exp_group_idx'] == 'v44':
        # load X
        version = f"v11/v{configs['d']}"
        data_path = os.path.join(data_dir, version, 'X', 'X_1.pkl')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        X, W_true = data['X'], data['W_true']
        B_true = (W_true != 0)

        # load W
        version = f"v39/W_{configs['d']}_{configs['s0']}_{configs['seed_W']}.pkl"
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
        G_true = nx.DiGraph(W_true)
        if configs['W_type'] == 'W_true':
            G = G_true
            W = W_true
        elif configs['W_type'] == 'W_est':
            G = G_est
            W = W_est

        # utility function
        def postprocess_res(res_list: list, self_n: int):
            res_list = list(set(res_list))
            if self_n in res_list:
                res_list.remove(self_n)
            return res_list

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

        # fit knockoff
        X_tilde = np.zeros_like(X)
        nodes = list(G.nodes())
        for j in tqdm(nodes):
            preds = 0.
            if configs['v44_option'] == 1:
                X_input = [
                    func(G, j, X) for func in [parents_X, children_X, parents_of_children_X]
                ]
            elif configs['v44_option'] == 2:
                X_input = [
                    func(G, j, X) for func in [children_X, parents_of_children_X]
                ]
            elif configs['v44_option'] == 3:
                X_input = ego_graph_X(G, j, X, configs['v44_option_3_radius'])

            elif configs['v44_option'] == 4:
                child_X, child = children_X(G, j, X, True)
                par_of_child_X, par_of_child = parents_of_children_X(G, j, X, True)
                if child is None and par_of_child is None:
                    preds = None
                elif child is None and par_of_child is not None:
                    preds = None
                elif par_of_child is None and child is not None:
                    X_input = child_X
                else:
                    _W = W[par_of_child, :][:, child]
                    X_input = child_X - par_of_child_X @ _W

            elif configs['v44_option'] == 5:
                par_X, par = parents_X(G, j, X, True)
                child_X, child = children_X(G, j, X, True)
                par_of_child_X, par_of_child = parents_of_children_X(G, j, X, True)

                if child is None and par_of_child is None:
                    X_input = None
                elif child is None and par_of_child is not None:
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
                        preds = None
                    else:
                        pass # X_input kept

            if preds is not None:
                if isinstance(X_input, list):
                    X_input = [val for val in X_input if val is not None]
                    X_input = np.concatenate(X_input, axis=1)
                preds = _get_single_clf(X_input, X[:, j], 
                                        configs['method_diagn_gen'], 
                                        configs['lasso_alpha'],
                                        configs['device'])
            else:
                preds = 0.
            residuals = X[:, j] - preds
            indices_ = np.arange(residuals.shape[0])
            np.random.shuffle(indices_)
            sample = preds + residuals[indices_]
            sample = _adjust_marginal(sample, X[:, j], discrete=False)
            X_tilde[:, j] = sample

        suffix = ""
        if configs['v44_option'] == 3:
            suffix += f'_radius_{configs["v44_option_3_radius"]}'
        if configs['disable_dag_control']:
            suffix += f'_disable_dag_control'

        data_dir = os.path.join(
            data_dir,
            configs["exp_group_idx"],
            f'v{configs["d"]}_{configs["W_type"]}_option_{configs["v44_option"]}_{configs["method_diagn_gen"]}_{configs["lasso_alpha"]}'+suffix,
            "knockoff"
        )
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        data_path = os.path.join(data_dir, f"knockoff_1.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump(X_tilde, f)

        config_path = os.path.join(data_dir, f"knockoff_1_configs.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(configs, f)

    elif configs['exp_group_idx'] == 'v45':
        # load X
        version = f"v11/v{configs['d']}"
        data_path = os.path.join(data_dir, version, 'X', 'X_1.pkl')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        X, W_true = data['X'], data['W_true']
        B_true = (W_true != 0)

        # load W
        version = f"v39/W_{configs['d']}_{configs['s0']}_{configs['seed_W']}.pkl"
        data_path = os.path.join(data_dir, version)
        with open(data_path, 'rb') as f:
            W_est = pickle.load(f)

        # preprocessing
        assert configs['W_type'] is None, "W is of no use"
        # if configs['disable_dag_control']:
        #     G_est = nx.DiGraph(W_est)
        # else:
        #     mask = utils.extract_dag_mask(np.abs(W_est), 0)
        #     W_est[~mask] = 0.
        #     G_est = nx.DiGraph(W_est)
        #     assert nx.is_directed_acyclic_graph(G_est)
        # G_true = nx.DiGraph(W_true)
        # if configs['W_type'] == 'W_true':
        #     G = G_true
        #     W = W_true
        # elif configs['W_type'] == 'W_est':
        #     G = G_est
        #     W = W_est
        if not configs["v45_disable_knockoff_fit"]:
            with open('./configs_dagma.yaml', 'r') as f:
                configs_all = yaml.safe_load(f)
            configs_all = utils.combine_configs(configs_all, configs)
            configs_all['seed_knockoff'] = configs_all['seed_X']

            # fit knockoff
            X_tilde = knockoff.knockoff(X, configs_all)
        else:
            X_tilde = X.copy()
        rng = np.random.default_rng(seed=configs['seed_X'])
        C = rng.normal(0, configs['v45_sigma'], X_tilde.shape)
        X_tilde += C

        suffix = ""
        if configs["lasso_alpha"] == 'lasso':
            suffix += f'_{configs["lasso_alpha"]}'
        if configs["v45_disable_knockoff_fit"]:
            suffix += f'_disable_knockoff_fit'


        data_dir = os.path.join(
            data_dir,
            configs["exp_group_idx"],
            f'v{configs["d"]}_{configs["method_diagn_gen"]}_sigma_{configs["v45_sigma"]}'+suffix,
            "knockoff"
        )
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        data_path = os.path.join(data_dir, f"knockoff_1.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump(X_tilde, f)

        config_path = os.path.join(data_dir, f"knockoff_1_configs.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(configs, f)
        

    print("DONE!")
        
        # elif configs['v42_ii_idx'] == 2:
            



                
