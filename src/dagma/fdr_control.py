
import numpy as np
import yaml
import utils_dagma as utils_dagma
import torch
import logging

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

logger = logging.getLogger(__name__)

def type_1_threshold(W : np.ndarray, B_true : np.ndarray):
    print("start FDR threshold")
    t_list = np.sort(np.unique(np.abs(W)))
    for t in t_list:
        D1, D2, D_D, pos = \
            W[num_feat:, :], W[:num_feat, num_feat:], W[num_feat:, num_feat:], W[:num_feat, :num_feat]
        D1[np.abs(D1) <= t], D2[np.abs(D2) <= t], D_D[np.abs(D_D) <= t], pos[np.abs(pos) <= t] = \
            0., 0., 0., 0.
        num_D = (D1 != 0).sum() + (D2 != 0).sum()
        num_D_D = (D_D != 0).sum()
        num_pos = (pos != 0).sum()
        _fdr = (num_D - num_D_D).astype('float') / num_pos.astype('float')
        print(f"t: {t} | fdr estimate: {_fdr}")

        _W = W[:num_feat, :num_feat].copy()
        _W[np.abs(_W) <= t] = 0.
        try:
            acc = utils_dagma.count_accuracy(B_true, _W != 0.)
        except ValueError as e:
            print(f"threshold {t} | {e}")
            continue
        print(f"threshold {t} | fdr {acc['fdr']}")
        print("end FDR threshold")
        # if _fdr <= fdr:
        #     return t, _fdr

def type_1_control(W : np.ndarray, W_true : np.ndarray, fdr : int):
    """
    W: a 2p * 2p block matrix, W = 
        [ T_T | T_D ]
        [ --- | --- ]
        [ D_T | D_D ]
    where T are original features and D are knockoff features

    W_true : p * p weighted directed adjacent matrix of the ground truth graph

    fdr: the expected FDR.
    """
    print(f"==============================")
    print(f"expected FDR {fdr}")
    t_list = np.sort(np.unique(np.abs(W[:num_feat, :num_feat])))
    t_last = t_list[-1]
    fdr_est_last = 0.
    has_update = False

    T_T, T_D, D_T, D_D = \
        W[:num_feat, :num_feat].copy(), W[:num_feat, num_feat:].copy(), \
        W[num_feat:, :num_feat].copy(), W[num_feat:, num_feat:].copy()
    T_T, T_D, D_T, D_D = np.abs(T_T), np.abs(T_D), np.abs(D_T), np.abs(D_D)
    
    # from the largest to the smallest possible threshold (t)
    ## with t decreasing, FDR goes up. we take the minimal t such that
    ## estimated FDR < expected FDR.
    for t in reversed(t_list): 
        # compute estimated FDR
        ## FDR = ( #D - #D_D ) / #T = ( #T_D + #D_T + #D_D - #D_D) / #T = ( #T_D + #D_T ) / #T
        num_D = (T_D >= t).sum() + (D_T >= t).sum()
        num_T = (T_T >= t).sum()
        fdr_est = num_D.astype('float') / num_T.astype('float')
        print(f"threshold {t:.4f} | estimate fdr {fdr_est:.4f} ")
        if fdr_est <= fdr:
            t_last = t
            fdr_est_last = fdr_est
            has_update = True
    
    T_T = W[:num_feat, :num_feat].copy()
    T_T = np.abs(T_T)
    mask = T_T >= t_last
    T_T[mask], T_T[~mask] = 1, 0
    T_T_true = np.abs(W_true)
    mask = T_T_true > 0.
    T_T_true[mask], T_T_true[~mask] = 1, 0
    try:
        fdr_true = utils_dagma.count_accuracy(T_T_true, T_T)['fdr']
    except Exception as e:
        print(e)

    fdr_est_last = fdr_est_last if has_update else 1.0
    print(f"expected fdr {fdr:.4f} | selected threshold {t_last:.4f} | estimated fdr {fdr_est_last:.4f} | ground truth fdr {fdr_true:.4f}")
    print(f"==============================")

def type_2_control(W : np.ndarray, W_true : np.ndarray, fdr : int):
    pass

def type_3_control(W : np.ndarray, W_true : np.ndarray, fdr : int, Z_true : torch.Tensor = None, Z_knock : torch.Tensor = None):
    logger.info(f"==============================")
    logger.info(f"expected FDR {fdr}")

    if Z_true is not None and Z_knock is not None:
        Z = np.abs(Z_true) - np.abs(Z_knock)    
    else:
        Z = np.abs(W[:num_feat, :]) - np.abs(W[num_feat:, :])

    if numeric:
        Z[(Z <= numeric_precision) & (Z >= -numeric_precision)] = 0.
    
    T_T_true = np.abs(W_true)
    mask = (T_T_true > 0.)
    T_T_true[mask], T_T_true[~mask] = 1, 0

    for i, col in enumerate(Z.T):
        if abs_t_list:
            t_list = np.concatenate(([0], np.sort(np.unique(np.abs(col)))))
        else:
            t_list = np.sort(np.concatenate(([0], np.unique(col))))
        fdr_est_last = 1.
        fdr_true_last, power_last = 1., 1.
        t_last = np.inf
            
        for t in reversed(t_list):
            if t < 0.:
                break
            if est_type == 'tau':
                fdr_est = ((col <= -t).sum()) / np.max((1, (col >= t).sum()))
            else:
                fdr_est = (1 + (col <= -t).sum()) / np.max((1, (col >= t).sum()))
            
            T_T = col.copy()
            mask = (T_T >= t)
            T_T[mask], T_T[~mask] = 1, 0
            perf = utils_dagma.count_accuracy(T_T_true[:, i], T_T, verify_dag = False)
            fdr_true, power = perf['fdr'], perf['tpr']

            logger.debug(f"feat {i} | thresh {t:.4f} | est fdr {fdr_est:.4f} | true fdr {fdr_true:.4f}")

            if fdr_est <= fdr:
                t_last = t
                fdr_est_last = fdr_est
                fdr_true_last, power_last = fdr_true, power

        logger.debug(f"end feat {i} | expected fdr {fdr:.4f} | sel thresh {t_last:.4e} | "
              f"est fdr {fdr_est_last:.4f} | true fdr {fdr_true_last:.4f} | true power {power_last:.4f}")
        if abs_selection:
            mask = (np.abs(Z[:, i] >= t_last))
        else:
            mask = (Z[:, i] >= t_last)
        Z[mask, i], Z[~mask, i] = 1, 0
    T_T = Z
    
    perf = utils_dagma.count_accuracy(T_T_true, T_T)
    fdr_true, power = perf['fdr'], perf['tpr']
    if utils_dagma.is_dag(T_T):
        logger.info("W_est is DAG")
    else:
        logger.info("W_est is NOT DAG")
    logger.info(f"expected fdr {fdr:.4f} | true fdr {fdr_true:.4f} | true power {power:.4f}")
    logger.info(f"==============================")
    return fdr_true, power

def type_3_control_global(W : np.ndarray, W_true : np.ndarray, fdr : int): # TODO: not finished
    logger.info(f"==============================")
    logger.info(f"expected FDR {fdr}")

    Z = np.abs(W[:num_feat, :]) - np.abs(W[num_feat:, :])

    if numeric:
        Z[(Z <= numeric_precision) & (Z >= -numeric_precision)] = 0.
    
    T_T_true = np.abs(W_true)
    mask = (T_T_true > 0.)
    T_T_true[mask], T_T_true[~mask] = 1, 0

    fdr_est_last = 1.
    t_last = np.inf

    if abs_t_list:
        t_list = np.concatenate(([0], np.sort(np.unique(np.abs(Z)))))
    else:
        t_list = np.sort(np.concatenate(([0], np.unique(Z))))

    for t in reversed(t_list):
        if t < 0.:
            break
        if est_type == 'tau':
            fdr_est = ((Z <= -t).sum()) / np.max((1, (Z >= t).sum()))
        else:
            fdr_est = (1 + (Z <= -t).sum()) / np.max((1, (Z >= t).sum()))
        
        T_T = Z.copy()
        mask = (T_T >= t)
        T_T[mask], T_T[~mask] = 1, 0
        perf = utils_dagma.count_accuracy(T_T_true, T_T)
        fdr_true, power = perf['fdr'], perf['tpr']

        logger.debug(f"thresh {t:.4f} | est fdr {fdr_est:.4f} | true fdr {fdr_true:.4f} | true power {power:.4f}")

        if fdr_est <= fdr:
            t_last = t
            fdr_est_last = fdr_est

    if abs_selection:
        mask = (np.abs(Z >= t_last))
    else:
        mask = (Z >= t_last)
    Z[mask], Z[~mask] = 1, 0
    T_T = Z
    
    perf = utils_dagma.count_accuracy(T_T_true, T_T)
    fdr_true, power = perf['fdr'], perf['tpr']

    if utils_dagma.is_dag(T_T):
        logger.info("W_est is DAG")
    else:
        logger.info("W_est is NOT DAG")
    logger.info(f"expected fdr {fdr:.1f} | sel thresh {t_last:.4e} | "
          f"est fdr {fdr_est_last:.4f} | true fdr {fdr_true:.4f} | true power {power:.4f}")
    logger.info(f"==============================")
    return fdr_true, power

    
