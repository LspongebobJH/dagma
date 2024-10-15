import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from tqdm import tqdm


def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in tqdm(range(max_iter)):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est_no_filter = W_est.copy()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est_no_filter, W_est


if __name__ == '__main__':
    import utils, os, pickle, utils_dagma
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--d', type=int, default=None)
    parser.add_argument('--s0', type=int, default=None)
    parser.add_argument('--seed_X', type=int, default=1)
    parser.add_argument('--src_note', type=str, default="")
    parser.add_argument('--dst_note', type=str, default="")
    parser.add_argument('--device', type=str, default='cuda:7') 

    # experimentally testing hyperparameters)
    args = parser.parse_args()
    utils.set_random_seed(0)
    
    # n, d = 2000, args.d
    # s0 = args.s0
    # version = f"v11/v{d}_{s0}" + args.src_note
    # device = args.device
    # root_dir = '/home/jiahang/dagma/src/dagma/simulated_data'
    # data_path = os.path.join(root_dir, version, 'X', f'X_{args.seed_X}.pkl')

    # with open(data_path, 'rb') as f:
    #     data = pickle.load(f)
    # X, W_true = data['X'], data['W_true']
    # B_true = (W_true != 0)
    # print("fit notears")

    n, d, s0, graph_type, sem_type = 1000, 40, 80, 'ER', 'gauss'
    B_true = utils_dagma.simulate_dag(d, s0, graph_type)
    W_true = utils_dagma.simulate_parameter(B_true)
    X = utils_dagma.simulate_linear_sem(W_true, n, sem_type)

    W_est_no_filter, W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
    acc = utils_dagma.count_accuracy(B_true, W_est != 0, use_logger=False)
    print(acc)

    prec, rec, threshold = precision_recall_curve(B_true.astype(int).flatten(), np.abs(W_est).flatten())
    auprc = auc(rec, prec)
    auroc = roc_auc_score(B_true.astype(int).flatten(), np.abs(W_est).flatten())

    prec_trunc, rec_trunc, threshold_trunc = \
        precision_recall_curve(B_true.astype(int).flatten(), 
                               np.abs(W_est).flatten())
    auprc_trunc = auc(rec_trunc, prec_trunc)
    auroc_trunc = roc_auc_score(B_true.astype(int).flatten(), 
                                np.abs(W_est).flatten())

    print(f"auprc: {auprc:.2f} | auroc: {auroc:.2f}")
    print(f"auprc_trunc: {auprc_trunc:.2f} | auroc_trunc: {auroc_trunc:.2f}")

    # data_dir = os.path.join(root_dir, "v39", f"{d}_{s0}")
    # if not os.path.exists(data_dir):
    #     os.makedirs(data_dir)
    # data_path = os.path.join(data_dir, f"W_{d}_{s0}_{args.seed_X}_0{args.dst_note}.pkl")
    # with open(data_path, 'wb') as f:
    #     pickle.dump(W_est_no_filter, f)
    # print("DONE")
