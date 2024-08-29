import numpy as np
import pickle
import os
import utils
import deconv
from fdr_control import (
    type_1_control, 
    type_3_control, 
    type_3_control_global,
    type_4_control_global,
    type_4_control,
)

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
with open('./configs_dagma.yaml', 'r') as f:
    configs = yaml.safe_load(f)

parser = ArgumentParser()
parser.add_argument('--n', type=int, default=None)
parser.add_argument('--d', type=int, default=None)
parser.add_argument('--s0', type=int, default=None)
parser.add_argument('--root_path', type=str, default=None)
parser.add_argument('--device', type=str, default=None)

parser.add_argument('--control_type', type=str, default='type_3', 
                    choices=['type_3', 'type_3_global', 'type_4', 'type_4_global'])
parser.add_argument('--dagma_type', type=str, default='dagma_1', 
                    choices = ['dagma_1'])
parser.add_argument('--dag_control', type=str, default=None)
parser.add_argument('--seed_X_list', type=str, required=True)
parser.add_argument('--seed_knockoff_list', type=str, required=True)
parser.add_argument('--seed_model_list', type=str, required=True)
parser.add_argument('--version', type=str, default=None)
parser.add_argument('--log_file', type=str, default='temp')
parser.add_argument('--n_jobs', type=int, default=1)

# testing trick for fdr control
parser.add_argument('--trick', type=str, default=None, 
                    choices=[None, 'trick_1', 'trick_2', 'trick_3', 'trick_3_1', 
                             'trick_4', 'trick_5', 'trick_6', 'trick_7', 'trick_8',
                             'trick_9', 'trick_10'])

# network deconvolution
parser.add_argument('--deconv_type', type=str, default=None, choices=[None, 'deconv_1', 'deconv_2'])
## valid only when deconv_1
parser.add_argument('--beta', type=float, default=None)
## valid only when deconv_2
parser.add_argument('--dag_control_deconv', type=str, default=None, choices=[None, 'dag_1'])
parser.add_argument('--alpha', type=float, default=None)
parser.add_argument('--abs_gt', action='store_true', default=None)
parser.add_argument('--order', type=int, default=None)


args = parser.parse_args()

configs = utils.combine_configs(configs, args)

"""
logging
"""
file_path = os.path.join('./logs', configs['log_file'])
parent_dir = os.path.dirname(file_path)
if not os.path.exists(parent_dir):
    os.makedirs(parent_dir, exist_ok=True)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename = file_path,
                    filemode='w')
# stdout and stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# brief
brief = logging.FileHandler(filename=os.path.join("./logs", f"{configs['log_file']}.brief"), mode='w')
brief.setLevel(logging.INFO)
brief.setFormatter(formatter)
logging.getLogger().addHandler(brief)

logger = logging.getLogger(__name__)


logger.info("configs")
logger.info("%s\n", pformat(configs, width=20))


if __name__ == '__main__':
        
    _configs = configs.copy()
    
    # type_3_global and type_3 have the same knockoff statistics, but different 
    ## FDR estimation
    # if configs['control_type'] == 'type_3_global':
    #     _configs['control_type'] = 'type_3'
    fdp_true_list, power_list = [], []
    fdr = 0.2
    for seed_X in configs['seed_X_list']:
        for seed_knockoff in configs['seed_knockoff_list']:
            for seed_model in configs['seed_model_list']:
                
                logger.info(f"X: {seed_X} | knockoff: {seed_knockoff} | model: {seed_model}")
                
                _configs['seed_X'] = seed_X
                _configs['seed_knockoff'] = seed_knockoff
                _configs['seed_model'] = seed_model

                _configs['gen_type'] = 'X'
                data_X, _ = utils.process_simulated_data(None, _configs, behavior='load')
                W_true = data_X['W_true']

                _configs['gen_type'] = 'W'
                data_W, W_configs = utils.process_simulated_data(None, _configs, behavior='load')

                W_est = data_W['W_est']
                
                # removing self loops in case previous steps forget it.
                real_p = W_est.shape[0]
                W_est[np.eye(real_p, real_p).astype(bool)] = 0.
                W_est[np.eye(real_p, real_p, k=real_p // 2).astype(bool)] = 0.
                W_est[np.eye(real_p, real_p, k=-real_p // 2).astype(bool)] = 0.

                if configs['deconv_type'] == 'deconv_1':
                    W_est = utils.net_deconv(W_est, configs)
                    # W_est[:configs['d'], :configs['d']] = utils.net_deconv(W_est[:configs['d'], :configs['d']], configs)
                elif configs['deconv_type'] == 'deconv_2':
                    W_est = deconv.net_deconv(W_est, configs)

                if configs['dagma_type'] == 'dagma_1':
                    W_est = W_est[:, :configs['d']]

                """
                FDR Control based on learned adjacent matrix
                """
                if configs['control_type'] == 'type_3':
                    fdp_true, power = type_3_control(configs, W_est, W_true, fdr)
                elif configs['control_type'] == 'type_3_global':
                    fdp_true, power = type_3_control_global(configs, W_est, W_true, fdr, W_full = data_W['W_est'])
                elif configs['control_type'] == 'type_4_global':
                    fdp_true, power = type_4_control_global(configs, W_est, W_true, fdr, W_full = data_W['W_est'])
                elif configs['control_type'] == 'type_4':
                    fdp_true, power = type_4_control(configs, W_est, W_true, fdr, W_full = data_W['W_est'])
                else:
                    raise Exception(f"{configs['control_type']} not implemented yet.")
                fdp_true_list.append(fdp_true)
                power_list.append(power)

    fdr_mean = np.mean(fdp_true_list)
    fdr_std = np.std(fdp_true_list)

    power_mean = np.mean(power_list)
    power_std = np.std(power_list)

    logger.info(f"expected fdr {fdr} | fdr {fdr_mean:.4f}±{fdr_std:.4f} | power {power_mean:.4f}±{power_std:.4f}")