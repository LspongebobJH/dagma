import numpy as np
import pickle
import os
import utils
import utils_dagma as utils_dagma

from argparse import ArgumentParser
import pprint
import yaml

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
parser.add_argument('--knock_type', type=str, default='knockoff_gan', 
                    choices=['permutation', 'knockoff_gan', 'deep_knockoff'])
parser.add_argument('--gen_type', type=str, required=True, choices=['X', 'knockoff', 'W'], 
                    help="X: generate X copies. knockoff: generate knockoff copies for each X.")
# Note that type_3_global has the same knockoff statistics as type_3, only the FDR estimate different
parser.add_argument('--dagma_type', type=str, default='dagma_1', 
                    choices = ['dagma_1'])
parser.add_argument('--seed_X', type=int, default=1)
parser.add_argument('--seed_knockoff', type=int, default=1)
parser.add_argument('--seed_model', type=int, default=0)
parser.add_argument('--version', type=int, default=1)

args = parser.parse_args()

n = configs['n']
d = configs['d']
s0 = configs['s0']
graph_type = configs['graph_type']
sem_type = configs['sem_type']
num_feat = d

knock_type = args.knock_type
gen_type = args.gen_type
dagma_type = args.dagma_type
version = args.version
seed_X, seed_knockoff, seed_model = args.seed_X, args.seed_knockoff, args.seed_model

assert seed_model == 0, "no need for random model seeds now."

configs = {
    'n': n, 'd': d, 's0': s0, 
    'graph_type': graph_type, 'sem_type': sem_type, 
    'knock_type': knock_type, 'gen_type': gen_type, 'dagma_type': dagma_type,
    'seed_X': seed_X, 'seed_knockoff': seed_knockoff, 'seed_model': seed_model,
    'version': version
}
print("configs")
pprint.PrettyPrinter(width=20).pprint(configs)
print()


if __name__ == '__main__':
    if gen_type == 'X':
        # only one X for now
        assert seed_X == 1
        utils.set_random_seed(seed_X)

        B_true = utils_dagma.simulate_dag(d, s0, graph_type)
        W_true = utils_dagma.simulate_parameter(B_true)
        X = utils_dagma.simulate_linear_sem(W_true, n, sem_type)
        data_X = {'X': X, 'W_true': W_true}

        utils.process_simulated_data(data_X, configs, behavior='save')

    elif gen_type == 'knockoff':
        
        assert seed_knockoff is not None

        _configs = configs.copy()
        _configs['gen_type'] = 'X' # hack, dirty codes
        data_X, _ = utils.process_simulated_data(None, _configs, behavior='load')
        X, W_true = data_X['X'], data_X['W_true']
        utils.set_random_seed(seed_knockoff)

        X_tilde = utils.knockoff(X, configs)
        
        utils.process_simulated_data(X_tilde, configs, behavior='save')

    else: # W

        assert seed_knockoff is not None and seed_model is not None

        _configs = configs.copy()
        _configs['gen_type'] = 'X' # hack, dirty codes
        data_X, _ = utils.process_simulated_data(None, _configs, behavior='load')
        X, W_true = data_X['X'], data_X['W_true']

        _configs = configs.copy()
        _configs['gen_type'] = 'knockoff' # hack, dirty codes
        X_tilde, _ = utils.process_simulated_data(None, _configs, behavior='load')

        utils.set_random_seed(seed_model)
        X_all = np.concatenate([X, X_tilde], axis=-1)

        W_est_no_filter, Z_true, Z_knock = utils.fit(X, X_all, configs)
        data_W = {
            'W_est': W_est_no_filter, 'Z_true': Z_true, 'Z_knock': Z_knock
        }
        
        utils.process_simulated_data(data_W, configs, behavior='save')
        
