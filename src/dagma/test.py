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
    parser.add_argument('--exp_group_idx', type=str, default=None, choices=['v40', 'v42', 'v43'])
    parser.add_argument('--v40_exp_idx', type=int, default=None, choices=[0])
    parser.add_argument('--v42_i_idx', type=int, default=None, choices=[1, 2])
    parser.add_argument('--v42_ii_idx', type=int, default=None, choices=[1, 2, 3])
    parser.add_argument('--v42_W_seed', type=int, default=1, choices=[1, 2])
    parser.add_argument('--v43_method', type=str, default='elastic', choices=['lasso', 'elastic'])
    parser.add_argument('--v43_disable_dag_control', action='store_true', default=False)
    parser.add_argument('--seed_X', type=int, default=1)
    parser.add_argument('--d', type=int, required=True)
    parser.add_argument('--s0', type=int, default=None)
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:7')

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
        version = f"v39/W_{configs['d']}_{configs['s0']}_{configs['v42_W_seed']}.pkl"
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
                    _get_single_clf_ko)(X_start, j, 'lasso') for j in tqdm(range(p))))
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

        data_dir = os.path.join(data_dir, f'v42/v{configs["v42_i_idx"]}_{configs["v42_ii_idx"]}/v{configs["d"]}_{configs["s0"]}_{configs["v42_W_seed"]}/knockoff')
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
        version = f"v39/W_{configs['d']}_{configs['s0']}_{configs['v42_W_seed']}.pkl"
        data_path = os.path.join(data_dir, version)
        with open(data_path, 'rb') as f:
            W_est = pickle.load(f)

        # preprocessing
        if configs['v43_disable_dag_control']:
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

        if configs["v43_disable_dag_control"]:
            data_dir = os.path.join(data_dir, f'v43/v{configs["d"]}_{configs["s0"]}_{configs["v42_W_seed"]}_{configs["v43_method"]}_disable_dag_control/knockoff')    
        else:
            data_dir = os.path.join(data_dir, f'v43/v{configs["d"]}_{configs["s0"]}_{configs["v42_W_seed"]}_{configs["v43_method"]}/knockoff')
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
            



                
