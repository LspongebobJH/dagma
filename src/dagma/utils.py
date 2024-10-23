import os
import numpy as np
import pickle
import random
import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import torch
import yaml
import logging
from argparse import ArgumentParser
from numpy.linalg import eigh, inv
from genie3 import GENIE3
from notears_cpu import notears_linear
from golem import golem


import utils_dagma

logger = logging.getLogger(__name__)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_default_dtype(torch.double)


def process_simulated_data(data, configs, behavior):
    assert behavior in ['save', 'load']
    if behavior == 'save':
        force_save = configs.get('force_save', False)
    if behavior == 'load':
        assert data is None

    gen_type = configs['gen_type']
    data_dir, _seed, path_config, path_data = get_data_path(configs)
    
    if behavior == 'save' and not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if behavior == 'save':
        logger.debug(f"save {gen_type} with seed {_seed} in {data_dir}")

        if os.path.exists(path_config) and not force_save:
            logger.debug(f"{path_config} already exists.")
            return
        with open(path_config, 'w') as f:
            yaml.dump(configs, f)

        if os.path.exists(path_data) and not force_save:
            logger.debug(f"{path_config} already exists.")
            return
        with open(path_data, 'wb') as f:
            pickle.dump(data, f)

    elif behavior == 'load':
        logger.debug(f"load {gen_type} with seed {_seed} in {data_dir}")

        with open(os.path.join(path_data), 'rb') as f:
            data = pickle.load(f)
        with open(os.path.join(path_config), 'r') as f:
            config = yaml.safe_load(f)
        return data, config

def get_data_path(configs : dict):
    gen_type = configs['gen_type']
    root_path = configs['root_path']
    version = configs['version']
    data_dir = os.path.join(root_path, f'v{version}', f'{gen_type}')

    if gen_type == 'X':
        _seed = configs['seed_X']
    elif gen_type == 'knockoff':
        _seed = f"{configs['seed_X']}_{configs['seed_knockoff']}"
    else: # W_torch
        _seed = f"{configs['seed_X']}_{configs['seed_knockoff']}_{configs['seed_model']}"

    path_config = os.path.join(data_dir, f'{gen_type}_{_seed}_configs.yaml')
    path_data = os.path.join(data_dir, f'{gen_type}_{_seed}.pkl')

    return data_dir, _seed, path_config, path_data



def fit(X_all, configs, original=False):
    gen_W = configs['gen_W']
    dagma_type = configs['dagma_type']
    warm_iter = configs['warm_iter']
    tune_params = configs['tune_params']

    assert dagma_type == 'dagma_1'

    W_est_no_filter, Z_true, Z_knock = \
        None, None, None
    if gen_W is None:
        from linear import DagmaLinear
        model = DagmaLinear(loss_type='l2', verbose=True)
        W_est_no_filter, _ = model.fit(dagma_type, X_all, lambda1=0.02, return_no_filter=True)
    elif gen_W == 'torch':
        d = configs['d']
        device = configs['device']
        deconv_type_dagma = configs['deconv_type_dagma']
        order = configs['order']
        alpha = configs['alpha']
        use_g_dir_loss = configs['use_g_dir_loss']
        disable_block_diag_removal = configs['disable_block_diag_removal']
        T = configs['T']

        if device == 'mps':
            dtype = torch.float32
        else:
            dtype = torch.double

        from dagma_torch import DagmaLinear, DagmaTorch
        eq_model = DagmaLinear(d=d, device=device, dtype=dtype, 
                               dagma_type=dagma_type, use_g_dir_loss=use_g_dir_loss,
                               deconv_type_dagma=deconv_type_dagma, order=order, alpha=alpha, 
                               disable_block_diag_removal=disable_block_diag_removal,
                               original=original).to(device)
        
        model = DagmaTorch(eq_model, device=device, verbose=True, dtype=dtype)
        
        W_est_no_filter, _  = model.fit(X_all, lambda1=0.02, lambda2=0.005, warm_iter=warm_iter, 
                                        T=T,
                                        return_no_filter=True)

    elif gen_W in ['grnboost2', 'geni3', 'L1+L2']:
        W_est_no_filter = GENIE3(X_all, 
                                 knock_genie3_type=configs['knock_genie3_type'], 
                                 nthreads=configs['nthreads'], 
                                 use_knockoff=True, 
                                 use_grnboost2=True if gen_W == 'grnboost2' else False,
                                 disable_remove_self=configs['disable_remove_self'],
                                 disable_norm=configs['disable_norm'],
                                 importance=configs['importance'],
                                 tune_params=tune_params,
                                 model_type='tree' if gen_W in ['genie3', 'grnboost2'] else gen_W)

    elif gen_W in ['notears']:
        W_est_no_filter, _ = notears_linear(X_all, lambda1=0.1, loss_type='l2')

    elif gen_W in ['golem']: # TODO: only EV, w/o NV
        device = configs['device']
        W_est_no_filter = golem(X_all, lambda_1=2e-2, lambda_2=5.0,
                  equal_variances=True, device=device)
    
    return W_est_no_filter, Z_true, Z_knock

def combine_configs(configs_yaml : dict, args):
    configs = {}
    for key, val in configs_yaml.items():
        configs[key] = val

    if isinstance(args, dict):
        _args = args
    else:
        _args = vars(args)

    for key, val in _args.items():
        if val in ["None", "none", -1]:
            _args[key] = None

    for key, val in configs.items():
        if val in ["None", "none", -1]:
            configs[key] = None

    for key, val in _args.items():
        if key in configs.keys():
            if val is not None:
                logger.debug(f"{key}:{_args[key]} will be updated by {val}")
            else:
                continue
        if key in ['seed_X_list', 'seed_knockoff_list', 'seed_model_list']:
            val = [int(s.strip()) for s in val.split(",")]
            if val[-1] == -1:
                val = list(range(val[0], val[1]+1))
        configs[key] = val
    return configs

def net_deconv(W_est: np.ndarray, configs: dict):
    """
    network deconvolution
    """
    beta = configs['beta']
    d = W_est.shape[0]
    W_est = (W_est + W_est.T) / 2
    eigval, eigvec = eigh(W_est)
    eigval_p_max, eigval_n_min = eigval[eigval >= 0.].max(), eigval[eigval < 0.].min()

    beta = 0.9
    m1, m2 = beta / ((1 - beta) * eigval_p_max), -beta / ((1 + beta) * eigval_n_min)
    alpha = min(m1, m2)
    eigval_dir = eigval / (1 / alpha + eigval)

    W_dir = eigvec @ np.diag(eigval_dir) @ eigvec.T

    """
    remove diagonal elements
    """
    W_dir[np.eye(d).astype(bool)] = 0.
    W_dir[np.eye(d, k = d // 2).astype(bool)] = 0.
    W_dir[np.eye(d, k = - d // 2).astype(bool)] = 0.

    return W_dir

def norm(X):
    max_abs_col = np.abs(X).max(axis=0).reshape(1, -1)
    X = X / (max_abs_col + 1e-8)
    assert (np.abs(X).max(axis=0) <= 1.).all()
    return X

def extract_dag_mask(A : np.ndarray, extract_type : int, pre_mask : np.ndarray = None, only_positive = False):
    """
    Assuming that the original graph is NOT dag. This func will not help to test whether
    the original graph is a dag.
    Here A can be knockoff statistic matrix Z, or q-value matrix Q
    extract_type:
    0: the removal is from the smallest to the largest until DAG is satisified
    1: the inclusion is from the largest to the smallest until DAG is not satisfied
    2: the removal is from the largest to the smallest until DAG is satisfied
    3: the inclusion is from the smallest to the largest until DAG is not satisfied
    4: edges from the largest to the smallest,
        if an edge violates DAG, dont include it.
    5: edges from the smallest to the largest,
        if an edge violates DAG, then remove it.

    hypothesis: if A == Z, extract_type == 0 / 1 (or 2 / 3) are the same results. but
    if A == Q, 0 / 1 (or 2 / 3) are different since q-value is not completely monotonic with number of edges
    """
    assert extract_type in [0, 1, 2, 3, 4, 5]
    a_list = np.sort(np.unique(A.flatten()))
    if only_positive:
        a_list = a_list[a_list > 0.]
    if pre_mask is None:
        pre_mask = np.full(A.shape, fill_value = True)

    if extract_type == 0:
        for a in a_list:
            _A = A.copy()
            mask = (_A >= a)
            _A[mask * pre_mask], _A[~(mask * pre_mask)] = 1, 0

            if utils_dagma.is_dag(_A):
                break

    elif extract_type == 1:
        a_list = np.flip(a_list)
        mask_last = np.full(A.shape, fill_value = True)
        for a in a_list:
            _A = A.copy()
            mask = (_A >= a)
            _A[mask * pre_mask], _A[~(mask * pre_mask)] = 1, 0

            if not utils_dagma.is_dag(_A):
                break
            mask_last = mask
        mask = mask_last

    elif extract_type == 2:
        a_list = np.flip(a_list)
        for a in a_list:
            _A = A.copy()
            mask = (_A <= a)
            _A[mask * pre_mask], _A[~(mask * pre_mask)] = 1, 0

            if utils_dagma.is_dag(_A):
                break

    elif extract_type == 3:
        mask_last = np.full(A.shape, fill_value = True)
        for a in a_list:
            _A = A.copy()
            mask = (_A <= a)
            _A[mask * pre_mask], _A[~(mask * pre_mask)] = 1, 0

            if not utils_dagma.is_dag(_A):
                break
            mask_last = mask
        mask = mask_last

    elif extract_type in [4, 5]:
        if extract_type == 4:
            a_list = np.flip(a_list)
        mask_last = np.full(A.shape, fill_value = True)
        for a in a_list:
            _A = A.copy()
            mask = (_A >= a)
            _A[mask_last * mask * pre_mask], _A[~(mask_last * mask * pre_mask)] = 1, 0 

            if not utils_dagma.is_dag(_A):
                mask_last[A == a] = False
        
        _A = np.zeros(A.shape)
        if only_positive:
            _A[mask_last & (A > 0.)] = 1.
        else:
            _A[mask_last] = 1.

        assert utils_dagma.is_dag(_A)
        mask = mask_last

    return mask