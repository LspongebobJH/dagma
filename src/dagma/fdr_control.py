
import numpy as np
import yaml
import utils_dagma as utils_dagma
import torch
import logging

from utils import extract_dag_mask
from joblib import Parallel, delayed
from tqdm import tqdm
from copy import deepcopy

logger = logging.getLogger(__name__)

def type_1_threshold(configs : dict, W : np.ndarray, B_true : np.ndarray):
    num_feat = configs['d']

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
            acc = utils_dagma.count_accuracy_simplify(B_true, _W != 0.)
        except ValueError as e:
            print(f"threshold {t} | {e}")
            continue
        print(f"threshold {t} | fdr {acc['fdr']}")
        print("end FDR threshold")
        # if _fdr <= fdr:
        #     return t, _fdr

def type_1_control(configs : dict, W : np.ndarray, W_true : np.ndarray, fdr : int):
    """
    W: a 2p * 2p block matrix, W = 
        [ T_T | T_D ]
        [ --- | --- ]
        [ D_T | D_D ]
    where T are original features and D are knockoff features

    W_true : p * p weighted directed adjacent matrix of the ground truth graph

    fdr: the expected FDR.
    """
    num_feat = configs['d']
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
        fdr_true = utils_dagma.count_accuracy_simplify(T_T_true, T_T)['fdr']
    except Exception as e:
        print(e)

    fdr_est_last = fdr_est_last if has_update else 1.0
    print(f"expected fdr {fdr:.4f} | selected threshold {t_last:.4f} | estimated fdr {fdr_est_last:.4f} | ground truth fdr {fdr_true:.4f}")
    print(f"==============================")

def type_2_control(W : np.ndarray, W_true : np.ndarray, fdr : int):
    pass

def type_3_control(configs : dict, W : np.ndarray, W_true : np.ndarray, fdr : float, Z_true : torch.Tensor = None, Z_knock : torch.Tensor = None):
    num_feat = configs['d']
    est_type = configs['est_type']
    numeric = configs['numeric']
    numeric_precision = eval(configs['numeric_precision'])
    abs_t_list = configs['abs_t_list']
    abs_selection = configs['abs_selection']

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
    T_T_res = np.zeros_like(T_T_true, dtype=int)

    for i, col in enumerate(Z.T):
        if abs_t_list:
            t_list = np.concatenate(([0], np.sort(np.unique(np.abs(col)))))
        else:
            t_list = np.sort(np.concatenate(([0], np.unique(col))))
        fdr_est_last = 1.
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
            perf = utils_dagma.count_accuracy_simplify(T_T_true[:, i], T_T, verify_dag = False)
            fdr_true, power = perf['fdr'], perf['tpr']

            logger.debug(f"feat {i} | thresh {t:.4f} | est fdr {fdr_est:.4f} | true fdr {fdr_true:.4f}")

            if fdr_est <= fdr:
                t_last = t
                fdr_est_last = fdr_est
        
        if abs_selection:
            mask = (np.abs(Z[:, i] >= t_last))
        else:
            mask = (Z[:, i] >= t_last)
        Z_B = np.zeros(Z.shape[0], dtype=int)
        Z_B[mask], Z_B[~mask] = 1, 0
        perf = utils_dagma.count_accuracy_simplify(T_T_true[:, i], Z_B, verify_dag = False)
        fdr_true, power = perf['fdr'], perf['tpr']

        logger.debug(f"end feat {i} | expected fdr {fdr:.4f} | sel thresh {t_last:.4e} | "
              f"est fdr {fdr_est_last:.4f} | true fdr {fdr_true:.4f} | true power {power:.4f}")
        T_T_res[:, i] = Z_B
    
    perf = utils_dagma.count_accuracy_simplify(T_T_true, T_T_res)
    fdr_true, power = perf['fdr'], perf['tpr']
    if utils_dagma.is_dag(T_T_res):
        logger.info("W_est is DAG")
    else:
        logger.info("W_est is NOT DAG")
    logger.info(f"expected fdr {fdr:.4f} | true fdr {fdr_true:.4f} | true power {power:.4f}")
    logger.info(f"==============================")
    return fdr_true, power

def dag_control_proc(configs: dict, W_full: np.ndarray = None, Z: np.ndarray = None, Q: np.ndarray = None, pre_mask: np.ndarray = None):
    # dag control on Q is deprecated
    rng = np.random.default_rng(0)
    dag_control, d = configs['dag_control'], configs['d']
    if dag_control == 'dag_7': # before W -> Z
        W_full = np.abs(W_full)
        mask = extract_dag_mask(W_full, 0)
        W_full[~mask] = 0
        W = W_full[:, :d]
        return W
    elif dag_control == 'dag_9':
        W = np.abs(W_full[:d, :d])
        selectors = rng.binomial(1, 0.5, [d, d])
        for i in range(d):
            for j in range(i, d):
                if W[i, j] > W[j, i]:
                    W_full[j, i] = 0.
                    W_full[j+d, i] = 0.
                elif W[i, j] < W[j, i]:
                    W_full[i, j] = 0.
                    W_full[i+d, j] = 0.
                else:
                    if selectors[i, j] == 1:
                        W_full[j, i] = 0.
                        W_full[j+d, i] = 0.
                    else:
                        W_full[i, j] = 0.
                        W_full[i+d, j] = 0.
        W = W_full[:, :d]
        return W

    elif dag_control == 'dag_12':
        W = np.abs(W_full[:d, :d])
        mask = extract_dag_mask(W, 4)
        mask = np.concatenate([mask, mask], axis=0)
        W = W_full[:, :d]
        W[~mask] = 0.
        return W

    elif dag_control == 'dag_13':
        W = np.abs(W_full[:d, :d])
        mask = extract_dag_mask(W, 5)
        mask = np.concatenate([mask, mask], axis=0)
        W = W_full[:, :d]
        W[~mask] = 0.
        return W

    # after W -> Z and before fdr control
    if dag_control == 'dag_1':
        Z_min = Z.min()
        mask = extract_dag_mask(Z, 0)
        Z[~mask] = Z_min
        return Z
    elif dag_control == 'dag_2':
        Z_min = Z.min()
        mask = extract_dag_mask(Z, 1)
        Z[~mask] = Z_min
        return Z
    elif dag_control == 'dag_8':
        _Z = Z.copy()
        selectors = rng.binomial(1, 0.5, [d, d])
        for i in range(d):
            for j in range(i, d):
                if _Z[i, j] > _Z[j, i]:
                    _Z[j, i] = 0.
                elif _Z[i, j] < _Z[j, i]:
                    _Z[i, j] = 0.
                else:
                    if selectors[i, j] == 1:
                        _Z[j, i] = 0.
                    else:
                        _Z[i, j] = 0.
        return _Z
    elif dag_control == 'dag_10_0':
        mask = extract_dag_mask(Z, 4, only_positive=True)
        _Z = Z.copy()
        Z[~mask] = 0.
        assert (_Z[_Z < 0.] == Z[Z < 0.]).all()
        return Z

    elif dag_control == 'dag_10_Z_min':
        Z_min = Z.min()
        mask = extract_dag_mask(Z, 4, only_positive=True)
        Z[~mask] = Z_min
        return Z

    elif dag_control == 'dag_11_0':
        mask = extract_dag_mask(Z, 5, only_positive=True)
        _Z = Z.copy()
        Z[~mask] = 0.
        assert (_Z[_Z < 0.] == Z[Z < 0.]).all()
        return Z

    elif dag_control == 'dag_11_Z_min':
        Z_min = Z.min()
        mask = extract_dag_mask(Z, 5, only_positive=True)
        Z[~mask] = Z_min
        return Z

    

    
    # after fdr control, mask is given by fdr control
    if dag_control == 'dag_3':
        _Z = Z.copy()
        dag_mask = extract_dag_mask(_Z, 0, pre_mask = pre_mask)
        return dag_mask
    elif dag_control == 'dag_4':
        _Z = Z.copy()
        dag_mask = extract_dag_mask(_Z, 1, pre_mask = pre_mask)
        return dag_mask
    elif dag_control == 'dag_5':
        _Q = Q.copy()
        dag_mask = extract_dag_mask(_Q, 2, pre_mask = pre_mask)
        return dag_mask
    elif dag_control == 'dag_6':
        _Q = Q.copy()
        dag_mask = extract_dag_mask(_Q, 3, pre_mask = pre_mask)
        return dag_mask
    

def trick_proc(W_full: np.ndarray, configs: dict):
    trick, num_feat = configs['trick'], configs['d']
    if trick == 'trick_1':
        Z1 = np.abs(W_full[:num_feat, num_feat:]) - np.abs(W_full[num_feat:, num_feat:])
        Z = Z - Z1
    elif trick == 'trick_2':
        Z1 = np.abs(W_full[num_feat:, :num_feat]) - np.abs(W_full[num_feat:, num_feat:])
        Z = Z - Z1
    elif trick == 'trick_3':
        Z1 = np.abs(W_full[:num_feat, num_feat:]) - np.abs(W_full[num_feat:, num_feat:])
        Z2 = np.abs(W_full[:num_feat, :num_feat]) - np.abs(W_full[num_feat:, num_feat:])
        Z = Z - Z1 + Z2
    elif trick == 'trick_3_1':
        Z = 2 * np.abs(W_full[:num_feat, :num_feat]) - \
            np.abs(W_full[num_feat:, :num_feat]) - \
            np.abs(W_full[:num_feat, num_feat:])
    elif trick == 'trick_4':
        Z = np.abs(W_full[:num_feat, :num_feat]) - np.abs(W_full[num_feat:, num_feat:])
    elif trick == 'trick_5':
        Z = (np.abs(W_full[:num_feat, num_feat:]) + np.abs(W_full[num_feat:, :num_feat])) - \
            np.abs(W_full[num_feat:, num_feat:])
    elif trick == 'trick_6':
        Z = (np.abs(W_full[:num_feat, num_feat:]) + np.abs(W_full[num_feat:, :num_feat])) - \
            2 * np.abs(W_full[num_feat:, num_feat:])
    elif trick == 'trick_7':
        Z = np.abs(W_full[:num_feat, :num_feat]) - np.abs(W_full[:num_feat, num_feat:])
    elif trick == 'trick_8':
        Z = np.abs(W_full[:num_feat, :num_feat]) + np.abs(W_full[num_feat:, num_feat:]) - \
            2 * np.abs(W_full[num_feat:, :num_feat])

    elif trick == 'trick_9':
        k = 2
        W11, W21 = W_full[:num_feat, :num_feat], W_full[num_feat:, :num_feat]
        W_est_pow = 2e-1 * np.linalg.matrix_power(W_full, k)
        W11_pow, W21_pow = W_est_pow[:num_feat, :num_feat], W_est_pow[num_feat:, :num_feat]
        W11_1 = np.abs(W11) - np.abs(W11_pow)
        W21_1 = np.abs(W21) - np.abs(W21_pow)
        Z = W11_1 - W21_1

    elif trick == 'trick_10':
        k = 2
        W11, W21 = W_full[:num_feat, :num_feat], W_full[num_feat:, :num_feat]
        W_est_pow = 1e-1 * np.linalg.matrix_power(W_full, k)
        W11_pow, W21_pow = W_est_pow[:num_feat, :num_feat], W_est_pow[num_feat:, :num_feat]
        coef = 1 - (np.abs(W11) - np.abs(W11).min()) / (np.abs(W11).max() - np.abs(W11).min())
        W11_1 = np.abs(W11) - coef * np.abs(W11_pow)
        W21_1 = np.abs(W21) - np.abs(W21_pow)
        Z = W11_1 - W21_1
    
    return Z

def type_3_control_global(configs : dict, W : np.ndarray, W_true : np.ndarray, fdr : float, W_full: np.ndarray = None):
    num_feat = configs['d']
    est_type = configs['est_type']
    numeric = configs['numeric']
    numeric_precision = eval(configs['numeric_precision'])
    abs_t_list = configs['abs_t_list']
    abs_selection = configs['abs_selection']
    n_jobs = configs['n_jobs']
    dag_control = configs['dag_control']
    trick = configs['trick']

    logger.info(f"==============================")
    logger.info(f"expected FDR {fdr}")

    if not (np.diag(W[:num_feat, :]) == 0.).all():
        logger.warning("W11 has non-zero diagonals, now forcibly remove self-loops.")
        W[:num_feat, :] = W[:num_feat, :] - np.diag(np.diag(W[:num_feat, :]))
    if not (np.diag(W[num_feat:, :]) == 0.).all():
        logger.warning("W21 has non-zero diagonals, now forcibly remove self-loops.")
        W[num_feat:, :] = W[num_feat:, :] - np.diag(np.diag(W[num_feat:, :]))

    if dag_control in ['dag_7', 'dag_9', 'dag_12', 'dag_13']:
        W = dag_control_proc(configs, W_full=W_full)

    Z = np.abs(W[:num_feat, :]) - np.abs(W[num_feat:, :])

    if dag_control in ['dag_1', 'dag_2', 'dag_8', 'dag_10_0', 'dag_10_Z_min', 'dag_11_0', 'dag_11_Z_min']:
        Z = dag_control_proc(configs, Z=Z)
    if trick is not None:
        Z = trick_proc(W_full, configs)

    if numeric:
        Z[(Z <= numeric_precision) & (Z >= -numeric_precision)] = 0.
    
    if abs_t_list:
        t_list = np.concatenate(([0], np.sort(np.unique(np.abs(Z)))))
    else:
        t_list = np.sort(np.concatenate(([0], np.unique(Z))))

    T_T_true = np.abs(W_true)
    mask = (T_T_true > 0.)
    T_T_true[mask], T_T_true[~mask] = 1, 0

    fdr_est_last = 1.
    t_last = np.inf

    def _get_t(t_list: list):
        t_last = np.inf
        fdr_est_last = None
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
            perf = utils_dagma.count_accuracy_simplify(T_T_true, T_T)
            fdr_true, power = perf['fdr'], perf['tpr']

            # logger.debug(f"thresh {t:.4f} | est fdr {fdr_est:.4f} | true fdr {fdr_true:.4f} | true power {power:.4f}")
            print(f"DEBUG: thresh {t:.4f} | est fdr {fdr_est:.4f} | true fdr {fdr_true:.4f} | true power {power:.4f}")

            if fdr_est <= fdr:
                t_last = t
                fdr_est_last = fdr_est
        return t_last, fdr_est_last
        
    interval = len(t_list) // n_jobs
    intervals = [(j * interval, (j+1) * interval) for j in range(n_jobs - 1)]
    intervals.append(
        ((n_jobs-1) * interval, len(t_list))
    )
    res = Parallel(n_jobs=n_jobs)(
            delayed(_get_t)(
                t_list[interval[0]:interval[1]]
            ) for interval in tqdm(intervals)
        )


    res = np.array([list(_res) for _res in res if not np.isinf(_res[0]) and _res[1] is not None])
    if len(res) > 0: # otherwise, no edge being selected
        t_last = res[:, 0].min()
        t_last_idx = np.argmin(res[:, 0])
        fdr_est_last = res[t_last_idx, 1]

    if abs_selection: # NOTE: deprecated
        mask = (np.abs(Z >= t_last))
    else:
        mask = (Z >= t_last)
    _Z = deepcopy(Z)
    _Z[mask], _Z[~mask] = 1, 0

    dag_mask = np.full(mask.shape, fill_value=True)
    if dag_control is None or utils_dagma.is_dag(Z):
        T_T = _Z
    elif dag_control in ['dag_3', 'dag_4']:
        _Z = deepcopy(Z)
        dag_mask = dag_control_proc(configs, Z=_Z, pre_mask=mask)
    Z[mask * dag_mask], Z[~(mask * dag_mask)] = 1, 0
    T_T = Z
    
    perf = utils_dagma.count_accuracy_simplify(T_T_true, T_T)
    fdr_true, power = perf['fdr'], perf['tpr']

    if utils_dagma.is_dag(T_T):
        logger.info("W_est is DAG")
    else:
        logger.info("W_est is NOT DAG")
    logger.info(f"expected fdr {fdr:.1f} | sel thresh {t_last:.4e} | "
          f"est fdr {fdr_est_last:.4f} | true fdr {fdr_true:.4f} | true power {power:.4f}")
    logger.info(f"==============================")
    return fdr_true, power

def type_4_control_global(configs : dict, W : np.ndarray, W_true : np.ndarray, fdr : int, W_full: np.ndarray = None):
    """
    deprecated, merged with type_3_control_global
    TODO: tiem profiling, especially extract_dag_mask, where topo sort is time-consuming. can try binary search.
    W_full: dag_7 is applied to the whole W11, W12, W21, W22, then W = [W11 || W21], rather than the input one.
    """
    num_feat = configs['d']
    est_type = configs['est_type']
    numeric = configs['numeric']
    numeric_precision = eval(configs['numeric_precision'])
    dag_control = configs['dag_control']
    abs_t_list = configs['abs_t_list']
    abs_selection = configs['abs_selection']

    logger.info(f"==============================")
    logger.info(f"expected FDR {fdr}")

    if dag_control == 'dag_7':
        W_full = np.abs(W_full)
        mask = extract_dag_mask(W_full, 0)
        W_full[~mask] = 0
        W = W_full[:, :configs['d']]

    Z = np.abs(W[:num_feat, :]) - np.abs(W[num_feat:, :])

    if numeric:
        Z[(Z <= numeric_precision) & (Z >= -numeric_precision)] = 0.
    
    T_T_true = np.abs(W_true)
    mask = (T_T_true > 0.)
    T_T_true[mask], T_T_true[~mask] = 1, 0

    fdr_est_last = 1.
    t_last = np.inf

    if dag_control == 'dag_1':
        Z_min = Z.min()
        mask = extract_dag_mask(Z, 0)
        Z[~mask] = Z_min
    elif dag_control == 'dag_2':
        Z_min = Z.min()
        mask = extract_dag_mask(Z, 1)
        Z[~mask] = Z_min
    
    if abs_t_list:
        t_list = np.concatenate(([0], np.sort(np.unique(np.abs(Z)))))
    else:
        t_list = np.sort(np.concatenate(([0], np.unique(Z))))

    Q = np.full(Z.shape, fill_value=1.)
    for t in reversed(t_list):
        if t < 0.:
            break
        if est_type == 'tau':
            fdr_est = ((Z <= -t).sum()) / np.max((1, (Z >= t).sum()))
        else:
            fdr_est = (1 + (Z <= -t).sum()) / np.max((1, (Z >= t).sum()))
        Q[Z == t] = fdr_est
        
        T_T = Z.copy()
        mask = (T_T >= t)
        T_T[mask], T_T[~mask] = 1, 0
        perf = utils_dagma.count_accuracy_simplify(T_T_true, T_T)
        fdr_true, power = perf['fdr'], perf['tpr']

        logger.debug(f"thresh {t:.4f} | est fdr {fdr_est:.4f} | true fdr {fdr_true:.4f} | true power {power:.4f}")

        if fdr_est <= fdr:
            t_last = t
            fdr_est_last = fdr_est

    if abs_selection:
        mask = (np.abs(Z >= t_last))
    else:
        mask = (Z >= t_last)
    _Z = Z.copy()
    _Z[mask], _Z[~mask] = 1, 0

    dag_mask = np.full(mask.shape, fill_value=True)
    if not utils_dagma.is_dag(_Z):
        if dag_control == 'dag_3':
            _Z = Z.copy()
            dag_mask = extract_dag_mask(_Z, 0, pre_mask = mask)
        elif dag_control == 'dag_4':
            _Z = Z.copy()
            dag_mask = extract_dag_mask(_Z, 1, pre_mask = mask)
        elif dag_control == 'dag_5':
            _Q = Q.copy()
            dag_mask = extract_dag_mask(_Q, 2, pre_mask = mask)
        elif dag_control == 'dag_6':
            _Q = Q.copy()
            dag_mask = extract_dag_mask(_Q, 3, pre_mask = mask)
    Z[mask * dag_mask], Z[~(mask * dag_mask)] = 1, 0
    T_T = Z

    perf = utils_dagma.count_accuracy_simplify(T_T_true, T_T)
    fdr_true, power = perf['fdr'], perf['tpr']

    if utils_dagma.is_dag(T_T):
        logger.info("W_est is DAG")
    else:
        logger.info("W_est is NOT DAG")
    logger.info(f"expected fdr {fdr:.1f} | sel thresh {t_last:.4e} | "
          f"est fdr {fdr_est_last:.4f} | true fdr {fdr_true:.4f} | true power {power:.4f}")
    logger.info(f"==============================")
    return fdr_true, power

def type_4_control(configs : dict, W : np.ndarray, W_true : np.ndarray, fdr : int, W_full : np.ndarray = None):
    num_feat = configs['d']
    est_type = configs['est_type']
    numeric = configs['numeric']
    numeric_precision = eval(configs['numeric_precision'])
    abs_t_list = configs['abs_t_list']
    abs_selection = configs['abs_selection']
    dag_control = configs['dag_control']

    logger.info(f"==============================")
    logger.info(f"expected FDR {fdr}")

    if dag_control == 'dag_7':
        W_full = np.abs(W_full)
        mask = extract_dag_mask(W_full, 0)
        W_full[~mask] = 0
        W = W_full[:, :configs['d']]

    Z = np.abs(W[:num_feat, :]) - np.abs(W[num_feat:, :])

    if numeric:
        Z[(Z <= numeric_precision) & (Z >= -numeric_precision)] = 0.
    
    T_T_true = np.abs(W_true)
    mask = (T_T_true > 0.)
    T_T_true[mask], T_T_true[~mask] = 1, 0

    if dag_control == 'dag_1':
        Z_min = Z.min()
        mask = extract_dag_mask(Z, 0)
        Z[~mask] = Z_min
    elif dag_control == 'dag_2':
        Z_min = Z.min()
        mask = extract_dag_mask(Z, 1)
        Z[~mask] = Z_min

    _Z = Z.copy()
    Q = np.full(Z.shape, fill_value=1.)
    mask_list = []
    for i, col in enumerate(_Z.T):
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

            Q[col == t, i] = fdr_est
            
            T_T = col.copy()
            mask = (T_T >= t)
            T_T[mask], T_T[~mask] = 1, 0
            perf = utils_dagma.count_accuracy_simplify(T_T_true[:, i], T_T, verify_dag = False)
            fdr_true, power = perf['fdr'], perf['tpr']

            logger.debug(f"feat {i} | thresh {t:.4f} | est fdr {fdr_est:.4f} | true fdr {fdr_true:.4f}")

            if fdr_est <= fdr:
                t_last = t
                fdr_est_last = fdr_est
                fdr_true_last, power_last = fdr_true, power

        logger.debug(f"end feat {i} | expected fdr {fdr:.4f} | sel thresh {t_last:.4e} | "
              f"est fdr {fdr_est_last:.4f} | true fdr {fdr_true_last:.4f} | true power {power_last:.4f}")
        if abs_selection:
            mask = (np.abs(_Z[:, i] >= t_last))
        else:
            mask = (_Z[:, i] >= t_last)
        mask_list.append(mask)
        _Z[mask, i], _Z[~mask, i] = 1, 0
    
    mask = np.stack(mask_list).T
    dag_mask = np.full(mask.shape, fill_value=True)

    if not utils_dagma.is_dag(_Z):
        if dag_control == 'dag_3':
            _Z = Z.copy()
            dag_mask = extract_dag_mask(_Z, 0, pre_mask = mask)
        elif dag_control == 'dag_4':
            _Z = Z.copy()
            dag_mask = extract_dag_mask(_Z, 1, pre_mask = mask)
        elif dag_control == 'dag_5':
            _Q = Q.copy()
            dag_mask = extract_dag_mask(_Q, 2, pre_mask = mask)
        elif dag_control == 'dag_6':
            _Q = Q.copy()
            dag_mask = extract_dag_mask(_Q, 3, pre_mask = mask)
    Z[mask * dag_mask], Z[~(mask * dag_mask)] = 1, 0
    T_T = Z
    
    perf = utils_dagma.count_accuracy_simplify(T_T_true, T_T)
    fdr_true, power = perf['fdr'], perf['tpr']
    if utils_dagma.is_dag(T_T):
        logger.info("W_est is DAG")
    else:
        logger.info("W_est is NOT DAG")
    logger.info(f"expected fdr {fdr:.4f} | true fdr {fdr_true:.4f} | true power {power:.4f}")
    logger.info(f"==============================")
    return fdr_true, power
