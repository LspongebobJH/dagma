import numpy as np
import pickle
import os
import utils
from fdr_control import type_1_control, type_3_control, type_3_control_global
import utils

from argparse import ArgumentParser
import pprint
from pprint import pformat
import yaml
import logging

"""
reading hyperparameters

n: number of samples
d: number of nodes
s0: expected number of edges (one directional)
graph_type: random graph generator, ['ER', 'SF', 'BP']
sem_type: noise model, ['gauss', 'exp', 'gumbel', 'uniform', 'logistic', 'poisson']
"""
with open('./configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)

n = configs['n']
d = configs['d']
s0 = configs['s0']
graph_type = configs['graph_type']
sem_type = configs['sem_type']
est_type = configs['est_type']
fdr = configs['fdr']
numeric = configs['numeric']
numeric_precision = eval(configs['numeric_precision'])
abs_t_list = configs['abs_t_list']
abs_selection = configs['abs_selection']
num_feat = d

parser = ArgumentParser()
parser.add_argument('--control_type', type=str, default='type_3', 
                    choices=['type_1', 'type_2', 'type_3', 'type_3_global', 'type_3_nn'])
parser.add_argument('--dagma_type', type=str, default='dagma_1', 
                    choices = ['dagma_1'])
parser.add_argument('--seed_X', type=int, default=1)
parser.add_argument('--seed_knockoff_list', type=str, required=True)
parser.add_argument('--seed_model_list', type=str, required=True)
parser.add_argument('--version', type=int, default=1)
parser.add_argument('--log_file', type=str, default='temp')

args = parser.parse_args()

control_type = args.control_type
dagma_type = args.dagma_type
seed_X = args.seed_X
seed_knockoff_list = [int(s.strip()) for s in args.seed_knockoff_list.split(",")]
seed_model_list = [int(s.strip()) for s in args.seed_model_list.split(",")]
version = args.version
log_file = args.log_file

assert control_type in ['type_3', 'type_3_global']

"""
logging
"""
# set up logging to file - see previous section for more details
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename = os.path.join('./logs', log_file),
                    filemode='w')
# stdout and stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# brief
brief = logging.FileHandler(filename=os.path.join('./logs', f'{log_file}.brief'), mode='w')
brief.setLevel(logging.INFO)
brief.setFormatter(formatter)
logging.getLogger().addHandler(brief)

logger = logging.getLogger(__name__)

configs = {
    'n': n, 'd': d, 's0': s0, 'graph_type': graph_type, 'sem_type': sem_type, 'fdr': fdr, 
    'numeric': numeric, 'numeric_precision': numeric_precision,
    'control_type': control_type, 'est_type': est_type, 'dagma_type': dagma_type,
    'seed_X': seed_X, 'seed_knockoff_list': seed_knockoff_list, 'seed_model_list': seed_model_list,
    'version': version, 'log_file': log_file, 'abs_t_list': abs_t_list, 'abs_selection': abs_selection
}

logger.info("configs")
logger.info("%s\n", pformat(configs, width=20))


if __name__ == '__main__':
        
    _configs = configs.copy()
    _configs['gen_type'] = 'X'
    # type_3_global and type_3 have the same knockoff statistics, but different 
    ## FDR estimation
    if control_type == 'type_3_global':
        _configs['control_type'] = 'type_3'

    data_X, _ = utils.process_simulated_data(None, _configs, behavior='load')
    W_true = data_X['W_true']

    fdp_true_list, power_list = [], []

    _configs['gen_type'] = 'W'
    fdr = 0.2
    for seed_knockoff in seed_knockoff_list:
        for seed_model in seed_model_list:
            
            logger.info(f"knockoff: {seed_knockoff} | model: {seed_model}")
            
            _configs['seed_knockoff'] = seed_knockoff
            _configs['seed_model'] = seed_model
            
            data_W, W_configs = utils.process_simulated_data(None, _configs, behavior='load')

            W_est = data_W['W_est']
            if dagma_type == 'dagma_1':
                W_est = W_est[:, :d]

            """
            FDR Control based on learned adjacent matrix
            """
            if control_type == 'type_3':
                fdp_true, power = type_3_control(W_est, W_true, fdr)
            elif control_type == 'type_3_global':
                fdp_true, power = type_3_control_global(W_est, W_true, fdr)
            else:
                raise Exception(f"{control_type} not implemented yet.")
            fdp_true_list.append(fdp_true)
            power_list.append(power)

    fdr_mean = np.mean(fdp_true_list)
    fdr_std = np.std(fdp_true_list)

    power_mean = np.mean(power_list)
    power_std = np.std(power_list)

    logger.info(f"expected fdr {fdr} | fdr {fdr_mean:.4f}±{fdr_std:.4f} | power {power_mean:.4f}±{power_std:.4f}")