import os
import numpy as np
import pickle
import random
import tensorflow as tf
import torch
import yaml
import logging

from linear import DagmaLinear
from knockoff_gan import KnockoffGAN
from deep_knockoff.machine import KnockoffMachine
from deep_knockoff.parameters import GetTrainingHyperParams, SetFullHyperParams
from deep_knockoff.gaussian import GaussianKnockoffs

logger = logging.getLogger(__name__)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_file_path(n, d, s0, knock_type, control_type, seed): # deprecated
    return os.path.join('/home/jiahang/knockoff/simulated_data', 
            f'{n}_{d}_{s0}_{knock_type}_{control_type}_seed_{seed}.pkl')

def process_simulated_data(data, configs, behavior):
    assert behavior in ['save', 'load']
    if behavior == 'load':
        assert data is None

    gen_type = configs['gen_type']
    root_path = 'simulated_data'
    version = configs['version']
    
    data_dir = os.path.join(root_path, f'v{version}', f'{gen_type}')

    if behavior == 'save' and not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    if gen_type == 'X':
        _seed = configs['seed_X']
    else: # knockoff or W
        _seed = str(configs['seed_knockoff']) if gen_type == 'knockoff' else f"{configs['seed_knockoff']}_{configs['seed_model']}"

   
    if behavior == 'save':
        logger.debug(f"save {gen_type} with seed {_seed} in {data_dir}")
        with open(os.path.join(data_dir, f'{gen_type}_{_seed}_configs.yaml'), 'w') as f:
            yaml.dump(configs, f)
        with open(os.path.join(data_dir, f'{gen_type}_{_seed}.pkl'), 'wb') as f:
            pickle.dump(data, f)
    elif behavior == 'load':
        logger.debug(f"load {gen_type} with seed {_seed} in {data_dir}")
        with open(os.path.join(data_dir, f'{gen_type}_{_seed}.pkl'), 'rb') as f:
            data = pickle.load(f)
        with open(os.path.join(data_dir, f'{gen_type}_{_seed}_configs.yaml'), 'r') as f:
            config = yaml.safe_load(f)
        return data, config
        
def knockoff(X : np.ndarray, configs):
    n = configs['n']
    d = configs['d']
    knock_type = configs['knock_type']
    if 'seed' in configs.keys():
        seed = configs['seed']
    else:
        seed = configs['seed_knockoff']

    if knock_type == 'permutation':
        X_tilde = np.zeros_like(X)
        for seed, col in enumerate(range(X.shape[1])):
            rng = np.random.default_rng(seed=seed)
            X_tilde[:, col] = rng.permutation(X[:, col])

    elif knock_type == 'knockoff_gan':
        # TODO: tensorflow fail running on GPU, but not slow now.
        X_tilde = KnockoffGAN(x_train = X, x_name = 'Normal')

    elif knock_type == 'deep_knockoff':
        SigmaHat = np.cov(X, rowvar=False)
        second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(X,0), method="sdp")
        corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)
        training_params = GetTrainingHyperParams('gaussian')
        pars = SetFullHyperParams(training_params, n, d, corr_g)
        
        checkpoint_name = "checkpoints/deep_knockoff/" + 'gaussian'
        logs_name = "logs/log_3_model"

        machine = KnockoffMachine(pars, checkpoint_name=checkpoint_name, logs_name=logs_name)

        print("fitting deep knockoff")
        machine.train(X)
        X_tilde = machine.generate(X)

    elif knock_type == 'standard': # not work yet
        mean = np.mean(X, axis=0)
        cov = np.cov(X.T)
        inv_cov = np.linalg.inv(cov)
        s = np.ones(d) # TODO: should be customized
        diag_s = np.diag(s)
        new_mean = (X - diag_s @ inv_cov @ X.T).T
        new_cov = 2 * diag_s - diag_s @ inv_cov @ diag_s
        X_tilde = np.random.multivariate_normal(new_mean, new_cov, size=n)

    return X_tilde

def fit(X, X_all, configs):
    dagma_type = configs['dagma_type']
    assert dagma_type == 'dagma_1'
    
    W_est_no_filter, Z_true, Z_knock = \
        None, None, None
    model = DagmaLinear(loss_type='l2', verbose=True)
    W_est_no_filter, _ = model.fit(dagma_type, X_all, lambda1=0.02, return_no_filter=True)

    return W_est_no_filter, Z_true, Z_knock
