from utils_notears import LBFGSBScipy, trace_expm
import torch
import torch.nn as nn
import numpy as np
import math
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from tqdm import tqdm


class NotearsLinear(nn.Module):
    def __init__(self, d, device, original=False):
        super(NotearsLinear, self).__init__()
        self.d = d
        self.original = original
        self.device = device
        self.W = nn.Parameter(
                torch.empty((self.d, self.d), device=self.device)
            )
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
    

    def forward(self, x):  # [n, d] -> [n, d]
        W = self.clean_diag()
        return x @ W

    def clean_diag(self):
        """
        w is of shape [d/2 + d/2, d/2 + d/2], that is, w is
            [w11 | w12]
            [----|----]
            [w21 | w22]
        where wij in d/2 * d/2 matrix
        set diagonal of wij to 0
        """
        w = self.W
        d = self.d
        if self.original:
            diag_mat = torch.diag(torch.diag(w, 0))
        else:
            diag_mat = \
                torch.diag(torch.diag(w, 0)) + \
                torch.diag(torch.diag(w, d // 2), d // 2) + \
                torch.diag(torch.diag(w, -d // 2), -d // 2)
        w_res = w - diag_mat
        return w_res
        

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.d
        # fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = self.W
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        h = trace_expm(A) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(d) + A / d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, d - 1)
        # h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        fc1_weight = self.W  # [j * m1, i]
        reg = torch.sum(fc1_weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.W.abs())
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        W = self.W.cpu().detach().numpy()  # [i, j]
        return W


def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


def dual_ascent_step(model, X, lambda1, lambda2, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    X_torch = torch.from_numpy(X)
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch)
            loss = squared_loss(X_hat, X_torch)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg
            primal_obj.backward()
            return primal_obj
        optimizer.step(closure)  # NOTE: updates model in-place
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
        print(f"{h_new:.4f}")
    alpha += rho * h_new
    return rho, alpha, h_new


def notears_fit(model: NotearsLinear,
                      X: np.ndarray,
                      lambda1: float = 0.01, # default is 0
                      lambda2: float = 0.01, # default is 0
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in tqdm(range(max_iter)):
        rho, alpha, h = dual_ascent_step(model, X, lambda1, lambda2,
                                         rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break
    W_est = model.fc1_to_adj()
    W_est_no_filter = W_est.copy()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est_no_filter, W_est


def main():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    import utils_dagma as ut
    ut.set_random_seed(123)

    n, d, s0, graph_type, sem_type = 200, 5, 9, 'ER', 'mim'
    B_true = ut.simulate_dag(d, s0, graph_type)
    np.savetxt('W_true.csv', B_true, delimiter=',')

    X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
    np.savetxt('X.csv', X, delimiter=',')

    model = NotearsLinear(dims=[d, 10, 1], bias=True)
    W_est = notears_fit(model, X, lambda1=0.01, lambda2=0.01)
    assert ut.is_dag(W_est)
    np.savetxt('W_est.csv', W_est, delimiter=',')
    acc = ut.count_accuracy(B_true, W_est != 0)
    print(acc)


if __name__ == '__main__':
    raise Exception("Don't run this, not tested.")
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
    torch.set_default_dtype(torch.double)
    utils.set_random_seed(0)
    
    n, d = 2000, args.d
    s0 = args.s0
    version = f"v11/v{d}_{s0}" + args.src_note
    device = args.device
    root_dir = '/home/jiahang/dagma/src/dagma/simulated_data'
    data_path = os.path.join(root_dir, version, 'X', f'X_{args.seed_X}.pkl')

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    X, W_true = data['X'], data['W_true']
    B_true = (W_true != 0)

    model = NotearsLinear(d=d, device=device, original=True).to(device)
    print("fit notears")

    W_est_no_filter, W_est = notears_fit(model, X)
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