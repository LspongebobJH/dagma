import numpy as np
import pickle
import os
import knockoff
import utils
import utils_dagma as utils_dagma

from argparse import ArgumentParser
import pprint
import yaml
import networkx as nx
from time import time

"""
n: number of samples
d: number of nodes
s0: expected number of edges (one directional)
graph_type: random graph generator, ['ER', 'SF', 'BP']
sem_type: noise model, ['gauss', 'exp', 'gumbel', 'uniform', 'logistic', 'poisson']
"""
with open('./configs_dagma.yaml', 'r') as f:
    configs = yaml.safe_load(f)

parser = ArgumentParser()
parser.add_argument('--n', type=int, default=None)
parser.add_argument('--d', type=int, default=None)
parser.add_argument('--s0', type=int, default=None)
parser.add_argument('--device', type=str, default=None)
parser.add_argument('--root_path', type=str, default=None)
parser.add_argument('--graph_type', type=str, default='ER', 
                    choices=['ER', 'SF'])
parser.add_argument('--knock_type', type=str, default=None, 
                    choices=['permutation', 'deep_knockoff', 
                             'knockoff_diagn'])
parser.add_argument('--gen_type', type=str, required=True, 
                    choices=['X', 'knockoff', 'W', 'W_torch', 'W_genie3', 'W_grnboost2',
                             'W_L1+L2', 'W_notears', 'W_golem', 'W_dag-gnn'])
parser.add_argument('--d1', type=int, default=None, help="lead to bipartite graph")
parser.add_argument('--d2', type=int, default=None, help="lead to bipartite graph")
parser.add_argument('--noise_scale_X', type=float, default=1., help="available only when gen_type == X")
parser.add_argument('--norm_data_gen', type=str, default=None, choices=['topo_col', 'B_1_col', 'sym_1'])

# Note that type_3_global has the same knockoff statistics as type_3, only the FDR estimate different
parser.add_argument('--dagma_type', type=str, default=None, 
                    choices = ['dagma_1'])
parser.add_argument('--seed_X', type=int, default=1)
parser.add_argument('--seed_knockoff', type=int, default=1)
parser.add_argument('--seed_model', type=int, default=0)
parser.add_argument('--version', type=str, default=None)
parser.add_argument('--force_save', action='store_true', default=False)

# parameters of knockoffGAN in knockoff generation
parser.add_argument('--niter', type=int, default=None)
parser.add_argument('--norm_knockoffGAN', action='store_true', default=None)

# parameters of knockoffDiagn in knockoff generation
parser.add_argument('--disable_adjust_marg', action='store_true', default=None)
parser.add_argument('--method_diagn_gen', type=str, default=None, choices=['dagma', 'lasso', 'xgb', 'elastic'])
parser.add_argument('--lasso_alpha', type=str, default=None, choices=['knockoff_diagn', 'sklearn', 'OLS'])

# parameters of damga
parser.add_argument('--norm_DAGMA', action='store_true', default=None) # deprecated
parser.add_argument('--norm', type=str, default=None, choices=['col', 'row'])
parser.add_argument('--disable_block_diag_removal', action='store_true', default=None)
parser.add_argument('--deconv_type_dagma', type=str, default=None, 
                    choices=[None, 'deconv_1', 'deconv_2', 'deconv_4',
                             'deconv_4_1', 'deconv_4_2'])
parser.add_argument('--order', type=int, default=None)
parser.add_argument('--alpha', type=float, default=None)
parser.add_argument('--warm_iter', type=int, default=None)
parser.add_argument('--use_g_dir_loss', action='store_true', default=None)
parser.add_argument('--T', type=int, default=None)

# parameters of genie3 and grnboost2 fitting W
parser.add_argument('--disable_remove_self', action='store_true', default=False)
parser.add_argument('--disable_norm', action='store_true', default=False)
parser.add_argument('--knock_genie3_type', type=str, default='unified', choices=['separate', 'unified'])
parser.add_argument('--nthreads', type=int, default=1)
parser.add_argument('--importance', type=str, default='original', 
                    choices=['original', 'permutation', 'tree-shap'])

# tune hyperparameters of genie3 and grnboost2 when fitting W
# tune hyperparameters of tree models
parser.add_argument('--ntrees', type=int, default=None)
parser.add_argument('--max_feat', type=float, default=None)
parser.add_argument('--max_sample', type=float, default=None)

# tune hyperparameters of elatiscnet
parser.add_argument('--elastic_alpha', type=float, default=None)
parser.add_argument('--elastic_l1_ratio', type=float, default=None)

# deprecated
parser.add_argument('--cond_thresh_X', type=float, default=None, help="available only when gen_type == X")

args = parser.parse_args()

if args.d is None:
    assert args.d1 is not None and args.d2 is not None
    args.d = args.d1 + args.d2

args.gen_W = None

if args.gen_type not in ['X', 'knockoff', 'W']:
    args.gen_type, args.gen_W = args.gen_type.split('_', 1)

configs = utils.combine_configs(configs, args)

# assert configs['seed_model'] == 0, "no need for random model seeds now."


print("configs")
pprint.PrettyPrinter(width=20).pprint(configs)
print()


if __name__ == '__main__':

    _, _, path_config, path_data = utils.get_data_path(configs)
    if (os.path.exists(path_config) or os.path.exists(path_data)) and not configs['force_save']:
        print(f"{path_config} or {path_data} already exists, jump.")
    
    else:
        if configs['gen_type'] != 'X':
            assert configs['cond_thresh_X'] is None

        if configs['gen_type'] == 'X':
            if args.d1 is None or args.d2 is None:
                # only one X for now
                # assert configs['seed_X'] == 1
                utils.set_random_seed(configs['seed_X'])

                B_true = utils_dagma.simulate_dag(configs['d'], configs['s0'], configs['graph_type'])
                if configs['norm_data_gen'] == 'sym_1':
                    W_true = utils_dagma.simulate_parameter(B_true, [(-1., 1.)])
                else:  
                    W_true = utils_dagma.simulate_parameter(B_true)
                if configs['norm_data_gen'] == 'B_1_col':
                    W_true /= (B_true.sum(axis=0, keepdims=True) + 1.)
                X = utils_dagma.simulate_linear_sem(W_true, configs['n'], configs['sem_type'], norm_data_gen=configs['norm_data_gen'], 
                                                    noise_scale=configs['noise_scale_X'])
                cond_X = np.linalg.cond(X).item()
                configs['real_cond_X'] = cond_X

                if (configs['cond_thresh_X'] is not None and cond_X < configs['cond_thresh_X']) \
                    or configs['cond_thresh_X'] is None:

                    data_X = {'X': X, 'W_true': W_true}

                    utils.process_simulated_data(data_X, configs, behavior='save')
                else:
                    print(f"cond_X {cond_X} > {configs['cond_thresh_X']}")

            else: # Bipartite graph, deprecated
                assert configs['d1'] is not None and configs['d2'] is not None

                # simulate bipartite graph
                utils.set_random_seed(configs['seed_X'])
                G_true = nx.bipartite.gnmk_random_graph(n=configs['d1'], m=configs['d2'], k=configs['s0'])
                B_true = nx.to_numpy_array(G_true, nodelist=list(range(configs['d1'])) + list(range(configs['d1'], configs['d1']+configs['d2'])))
                B_true = np.triu(B_true)
                
                # simulate W_true
                if configs['norm_data_gen'] == 'sym_1':
                    W_true = utils_dagma.simulate_parameter(B_true, [(-1., 1.)])
                else:  
                    W_true = utils_dagma.simulate_parameter(B_true)
                if configs['norm_data_gen'] == 'B_1_col':
                    W_true /= (B_true.sum(axis=0, keepdims=True) + 1.)

                # simulated X
                X = utils_dagma.simulate_linear_sem(W_true, configs['n'], configs['sem_type'], norm_data_gen=configs['norm_data_gen'], 
                                                    noise_scale=configs['noise_scale_X'])
                cond_X = np.linalg.cond(X).item()
                configs['real_cond_X'] = cond_X

                # save data
                if (configs['cond_thresh_X'] is not None and cond_X < configs['cond_thresh_X']) \
                    or configs['cond_thresh_X'] is None:

                    data_X = {'X': X, 'W_true': W_true}

                    utils.process_simulated_data(data_X, configs, behavior='save')
                else:
                    print(f"cond_X {cond_X} > {configs['cond_thresh_X']}")

        elif configs['gen_type'] == 'knockoff': # deprecated
            
            assert configs['seed_knockoff'] is not None

            _configs = configs.copy()
            _configs['gen_type'] = 'X' # hack, dirty codes
            data_X, _ = utils.process_simulated_data(None, _configs, behavior='load')
            X, W_true = data_X['X'], data_X['W_true']
            utils.set_random_seed(configs['seed_knockoff'])

            X_tilde = knockoff.knockoff(X, configs)
            
            utils.process_simulated_data(X_tilde, configs, behavior='save')

        else: # [W, W_torch]

            time_st = time()
            assert configs['seed_knockoff'] is not None and configs['seed_model'] is not None

            _configs = configs.copy()
            _configs['gen_type'] = 'X' # hack, dirty codes
            data_X, _ = utils.process_simulated_data(None, _configs, behavior='load')
            X, W_true = data_X['X'], data_X['W_true']

            _configs = configs.copy()
            _configs['gen_type'] = 'knockoff' # hack, dirty codes
            X_tilde, _ = utils.process_simulated_data(None, _configs, behavior='load')

            utils.set_random_seed(configs['seed_model'])

            if configs['norm'] is not None:
                if configs['norm'] == 'col':
                    X_mean = X.mean(axis=0, keepdims=True)
                    X_std = X.std(axis=0, keepdims=True)
                elif configs['norm'] == 'row':
                    X_mean = X.mean(axis=1, keepdims=True)
                    X_std = X.std(axis=1, keepdims=True)
                X = (X - X_mean) / (X_std + 1e-8)

            configs['tune_params'] = {
                'ntrees': args.ntrees,
                'max_feat': args.max_feat,
                'max_sample': args.max_feat,

                'alpha': args.elastic_alpha,
                'l1_ratio': args.elastic_l1_ratio
            }

            if configs['gen_W'] in ['genie3', 'grnboost2'] and configs['knock_genie3_type'] == 'separate':
                X_all = {
                    'X': X,
                    'X_tilde': X_tilde
                }
            else:
                X_all = np.concatenate([X, X_tilde], axis=-1)

            W_est_no_filter, Z_true, Z_knock = utils.fit(X_all, configs)
            data_W = {
                'W_est': W_est_no_filter, 'Z_true': Z_true, 'Z_knock': Z_knock
            }
            
            utils.process_simulated_data(data_W, configs, behavior='save')
            print(f"time: {time() - time_st:.2f}s")
