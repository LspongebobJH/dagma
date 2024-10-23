import os
import numpy as np
import torch
import torch.optim as optim
from torch import nn
import utils_dagma
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from time import time

class GolemModel(nn.Module):
    """Set up the objective function of GOLEM.

    Hyperparameters:
        (1) GOLEM-NV: equal_variances=False, lambda_1=2e-3, lambda_2=5.0.
        (2) GOLEM-EV: equal_variances=True, lambda_1=2e-2, lambda_2=5.0.
    """

    def __init__(self, n, d, lambda_1, lambda_2, equal_variances=True,
                 B_init=None):
        """Initialize self.

        Args:
            n (int): Number of samples.
            d (int): Number of nodes.
            lambda_1 (float): Coefficient of L1 penalty.
            lambda_2 (float): Coefficient of DAG penalty.
            equal_variances (bool): Whether to assume equal noise variances
                for likelihood objective. Default: True.
            B_init (torch.Tensor or None): [d, d] weighted matrix for
                initialization. Set to None to disable. Default: None.
        """
        super(GolemModel, self).__init__()
        self.n = n
        self.d = d
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.equal_variances = equal_variances
        
        if B_init is not None:
            self.B = nn.Parameter(B_init.float())
        else:
            self.B = nn.Parameter(torch.zeros(d, d))

    def forward(self, X):
        """Forward pass of the model."""
        B = self._preprocess(self.B)
        likelihood = self._compute_likelihood(X, B)
        L1_penalty = self._compute_L1_penalty(B)
        h = self._compute_h(B)
        score = likelihood + self.lambda_1 * L1_penalty + self.lambda_2 * h
        return score, likelihood, h

    def _preprocess(self, B):
        """Set the diagonals of B to zero.

        Args:
            B (torch.Tensor): [d, d] weighted matrix.

        Returns:
            torch.Tensor: [d, d] weighted matrix.
        """
        B = B - torch.diag(torch.diag(B, 0))
        return B
        # return B.fill_diagonal_(0)

    def _compute_likelihood(self, X, B):
        """Compute (negative log) likelihood in the linear Gaussian case.

        Returns:
            torch.Tensor: Likelihood term (scalar-valued).
        """
        I = torch.eye(self.d).to(B.device)
        if self.equal_variances:    # Assuming equal noise variances
            return 0.5 * self.d * torch.log(
                torch.square(
                    torch.norm(X - X @ B)
                )
            ) - torch.logdet(I - B)
        else:    # Assuming non-equal noise variances
            return 0.5 * torch.sum(
                torch.log(
                    torch.sum(
                        torch.square(X - X @ B), dim=0
                    )
                )
            ) - torch.logdet(I - B)

    def _compute_L1_penalty(self, B):
        """Compute L1 penalty.

        Returns:
            torch.Tensor: L1 penalty term (scalar-valued).
        """
        return torch.norm(B, p=1)

    def _compute_h(self, B):
        """Compute DAG penalty.

        Returns:
            torch.Tensor: DAG penalty term (scalar-valued).
        """
        return torch.trace(torch.matrix_exp(B * B)) - self.d

def threshold_till_dag(B):
    """Remove the edges with smallest absolute weight until a DAG is obtained.

    Args:
        B (numpy.ndarray): [d, d] weighted matrix.

    Returns:
        numpy.ndarray: [d, d] weighted matrix of DAG.
        float: Minimum threshold to obtain DAG.
    """
    if utils_dagma.is_dag(B):
        return B, 0

    B = np.copy(B)
    # Get the indices with non-zero weight
    nonzero_indices = np.where(B != 0)
    # Each element in the list is a tuple (weight, j, i)
    weight_indices_ls = list(zip(B[nonzero_indices],
                                 nonzero_indices[0],
                                 nonzero_indices[1]))
    # Sort based on absolute weight
    sorted_weight_indices_ls = sorted(weight_indices_ls, key=lambda tup: abs(tup[0]))

    for weight, j, i in sorted_weight_indices_ls:
        if utils_dagma.is_dag(B):
            # A DAG is found
            break

        # Remove edge with smallest absolute weight
        B[j, i] = 0
        dag_thres = abs(weight)

    return B, dag_thres

def postprocess(B, graph_thres=0.3):
    """Post-process estimated solution:
        (1) Thresholding.
        (2) Remove the edges with smallest absolute weight until a DAG
            is obtained.

    Args:
        B (numpy.ndarray): [d, d] weighted matrix.
        graph_thres (float): Threshold for weighted matrix. Default: 0.3.

    Returns:
        numpy.ndarray: [d, d] weighted matrix of DAG.
    """
    B = np.copy(B)
    B[np.abs(B) <= graph_thres] = 0    # Thresholding
    B, _ = threshold_till_dag(B)

    return B

def train_iter(model, X, optimizer):
    """Training for one iteration.

    Args:
        model (GolemModel object): GolemModel.
        X (torch.Tensor): [n, d] data matrix.
        optimizer (torch.optim.Optimizer): Optimizer for the model.

    Returns:
        float: value of score function.
        float: value of likelihood function.
        float: value of DAG penalty.
        torch.Tensor: [d, d] estimated weighted matrix.
    """
    model.train()
    optimizer.zero_grad()
    score, likelihood, h = model(X)
    score.backward()
    optimizer.step()

    return score.item(), likelihood.item(), h.item(), model.B.detach()

def golem(X, lambda_1, lambda_2, equal_variances=True,
          num_iter=1e+5, learning_rate=1e-3,
          B_init=None, device='cpu'):
    """Solve the unconstrained optimization problem of GOLEM, which involves
        GolemModel and GolemTrainer.

    Args:
        X (numpy.ndarray): [n, d] data matrix.
        lambda_1 (float): Coefficient of L1 penalty.
        lambda_2 (float): Coefficient of DAG penalty.
        equal_variances (bool): Whether to assume equal noise variances
            for likelibood objective. Default: True.
        num_iter (int): Number of iterations for training.
        learning_rate (float): Learning rate of Adam optimizer. Default: 1e-3.
        checkpoint_iter (int): Number of iterations between each checkpoint.
            Set to None to disable. Default: None.
        output_dir (str): Output directory to save training outputs.
        B_init (numpy.ndarray or None): [d, d] weighted matrix for initialization.
            Set to None to disable. Default: None.

    Returns:
        numpy.ndarray: [d, d] estimated weighted matrix.

    Hyperparameters:
        (1) GOLEM-NV: equal_variances=False, lambda_1=2e-3, lambda_2=5.0.
        (2) GOLEM-EV: equal_variances=True, lambda_1=2e-2, lambda_2=5.0.
    """
    # Center the data
    X = X - X.mean(axis=0, keepdims=True)

    # Set up model
    n, d = X.shape
    model = GolemModel(n, d, lambda_1, lambda_2, equal_variances, B_init).to(device)

    # Training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    X = torch.tensor(X).to(device)
    score_last = 0.
    for i in range(0, int(num_iter) + 1):
        score, likelihood, h, B_est = train_iter(model, X, optimizer)
        print(f"iter {i} | score {score:.4f} | likelihood: {likelihood:.4f} | h: {h:.4f}")

        if abs(score - score_last) < 1e-6:
            print(f"early stop at iter {i}")
            break

        score_last = score

    return B_est.cpu().numpy()   # Not thresholded yet


if __name__ == '__main__':
    import utils, os, utils_dagma

    # Reproducibility
    utils.set_random_seed(0)
    torch.set_default_dtype(torch.double)

    # Load dataset
    n, d = 2000, 60
    graph_type, s0 = 'ER', 360    # ER2 graph
    sem_type = 'gauss'
    device = "cuda:7"
    # B_scale = 1.0
    # noise_type = 'gaussian_ev'
    # dataset = SyntheticDataset(n, d, graph_type, degree,
    #                            noise_type, B_scale)
    B_true = utils_dagma.simulate_dag(d, s0, graph_type)
    W_true = utils_dagma.simulate_parameter(B_true, [(-1., 1.)])
    X = utils_dagma.simulate_linear_sem(W_true, n, sem_type)

    # GOLEM-EV
    time_st = time()
    W_est_no_filter = golem(X, lambda_1=2e-2, lambda_2=5.0,
                  equal_variances=True, device=device)
    print(f"running time: {time() - time_st:.2f}s")

    # Post-process estimated solution and compute results
    W_est = postprocess(W_est_no_filter, graph_thres=0.3)
    results = utils_dagma.count_accuracy(B_true, W_est != 0, use_logger=False)
    print("Results (after post-processing): {}.".format(results))

    prec, rec, threshold = precision_recall_curve(B_true.astype(int).flatten(), np.abs(W_est).flatten())
    auprc = auc(rec, prec)
    auroc = roc_auc_score(B_true.astype(int).flatten(), np.abs(W_est).flatten())
    
    print(f"auprc: {auprc:.2f} | auroc: {auroc:.2f}")

    prec, rec, threshold = precision_recall_curve(B_true.astype(int).flatten(), np.abs(W_est_no_filter).flatten())
    fdp = 1. - prec
    target_fdp_thresh = np.where(fdp[-1:0:-1] < 0.2)[0][-1]
    power = rec[-1:0:-1][target_fdp_thresh]

    print(f"Given fdp < 0.2, the maximal power is {power}")