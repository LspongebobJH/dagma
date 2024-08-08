from locally_connected import LocallyConnected
import torch
import torch.nn as nn
import numpy as np
from  torch import optim
import copy
from tqdm.auto import tqdm
import typing
import math
from time import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.optim import Adam, lr_scheduler

__all__ = ["DagmaTorch"]

# TODO: rescaling W_pred to make itself converge [ deconv_1 X , deconv_2 Y ]

class DagmaLinear(nn.Module): 
    """
    Class that models the structural equations for the linear causal graph.
    """
    
    def __init__(self, d : int, 
                 dagma_type: str,
                 deconv_type_dagma: str = None,
                 order: int = None, alpha: float = None, use_g_dir_loss: bool = None,
                 disable_block_diag_removal : bool = None,
                 device : str = 'cuda:7', dtype: torch.dtype = torch.double, 
                 original : bool = False):
        r"""
        Parameters
        ----------
        dims : typing.List[int]
            Number of neurons in hidden layers of each MLP representing each structural equation.
        bias : bool, optional
            Flag whether to consider bias or not, by default ``True``
        dtype : torch.dtype, optional
            Float precision, by default ``torch.double``
        """
        torch.set_default_dtype(dtype)
        self.original = original
        self.disable_block_diag_removal = disable_block_diag_removal

        if self.original:
            self.d = d
        else:
            self.d = 2 * d
        self.device = device
        self.dagma_type = dagma_type
        self.deconv = deconv_type_dagma
        self.order = order
        self.alpha = alpha
        self.use_g_dir_loss = use_g_dir_loss
        
        super(DagmaLinear, self).__init__()
        self.I = torch.eye(self.d).to(self.device)

        if self.deconv in ['deconv_1', 'deconv_2', None]:
            self.W = Parameter(
                torch.empty((self.d, self.d), device=self.device)
            )
            init.kaiming_uniform_(self.W, a=math.sqrt(5))
            self.zero_mat = torch.zeros_like(self.W, device=self.device)
        elif self.deconv in ['deconv_3', 'deconv_4', 'deconv_4_1', 'deconv_4_2']:
            self.W1 = Parameter(
                torch.empty((self.d, self.d), device=self.device)
            )
            init.kaiming_uniform_(self.W1, a=math.sqrt(5))
            self.W2 = Parameter(
                torch.empty((self.d, self.d), device=self.device)
            )
            init.kaiming_uniform_(self.W2, a=math.sqrt(5))
            self.zero_mat = torch.zeros_like(self.W1, device=self.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n, d] -> [n, d]

        """
        remove diagonal
        """
        if self.deconv in ['deconv_1', 'deconv_2', None]:
            if self.dagma_type == 'dagma_1':
                W = self.clean_diag()
            else:
                W = self.W
        else:
            if self.dagma_type == 'dagma_1':
                W1, W2 = self.clean_diag()
            else:
                W1, W2 = self.W1, self.W2

        """
        forward (and deconv)
        """
        if self.deconv == 'deconv_1':
            W_cum = []
            W_cum.append(W)
            for i in range(1, self.order):
                W_cum.append(W_cum[i-1] @ W)
            W_cum = torch.stack(W_cum)
            regressor = W_cum.sum(dim=0)
        
        elif self.deconv == 'deconv_2':
            # note that eigen norm here is necessary to ensure that (I-W)^-1 is well defined.
            eigval = torch.linalg.eigvals(W.detach()).abs().max() # this eigval is for norm
            W = W / (eigval + 1e-8)
            regressor = torch.inverse(self.I - W) - self.I

        elif self.deconv == 'deconv_3':
            regressor = W1 * W1 - W2 * W2

        elif self.deconv == 'deconv_4':
            W1, W2 = self.remove_negative(W1, W2)
            regressor = W1 - W2

        elif self.deconv == 'deconv_4_1':
            W1, W2 = self.remove_negative(W1, W2)

            eigval1, eigval2 = \
                torch.linalg.eigvals(W1.detach()).abs().max(), \
                torch.linalg.eigvals(W2.detach()).abs().max()
            
            W1, W2 = \
                W1 / (eigval1 + 1e-8), \
                W2 / (eigval2 + 1e-8)
            regressor = 0.
            W1_pow, W2_pow = W1, W2
            for i in range(1, self.order + 1):
                regressor += (self.alpha ** i) * (W1_pow - W2_pow)
                W1_pow, W2_pow = W1_pow @ W1, W2_pow @ W2

        elif self.deconv == 'deconv_4_2':
            W1, W2 = self.remove_negative(W1, W2)

            eigval1, eigval2 = \
                torch.linalg.eigvals(W1.detach()).abs().max(), \
                torch.linalg.eigvals(W2.detach()).abs().max()
            W1, W2 = \
                self.W1 / (eigval1 + 1e-8), \
                self.W2 / (eigval2 + 1e-8)
            regressor = torch.inverse(self.I - self.alpha * W1) - torch.inverse(self.I - self.alpha * W2)

        else:
            regressor = self.W

        res = x @ regressor
        return res, regressor

    def h_func(self, regressor: torch.Tensor, s: float = 1.0) -> torch.Tensor:
        A = regressor ** 2  # [i, j]
        h = -torch.slogdet(s * self.I - A)[1] + self.d * np.log(s)
        return h

    def l1_reg(self, regressor) -> torch.Tensor:
        if self.deconv in ['deconv_1', 'deconv_2']:
            return torch.sum(torch.abs(self.W))
        elif self.deconv in ['deconv_3', 'deconv_4', 'deconv_4_1', 'deconv_4_2']:
            return torch.sum(torch.abs(self.W1)) + torch.sum(torch.abs(self.W2))
        return torch.sum(torch.abs(regressor))

    def get_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        if self.deconv in ['deconv_1', 'deconv_2', None]:
            if self.dagma_type == 'dagma_1':
                W = self.clean_diag()
            else:
                W = self.W
        else:
            if self.dagma_type == 'dagma_1':
                W1, W2 = self.clean_diag()
            else:
                W1, W2 = self.W1, self.W2

        if self.deconv in ['deconv_1', 'deconv_2', None]:
            return W
        elif self.deconv == 'deconv_3':
            return (W1 * W2 - W2 * W2)
        elif self.deconv in ['deconv_4', 'deconv_4_1', 'deconv_4_2']:
            W1, W2 = self.remove_negative(W1, W2)
            return W1 - W2
        
    def clean_diag(self):
        """
        w is of shape [d/2 + d/2, d/2 + d/2], that is, w is
            [w11 | w12]
            [----|----]
            [w21 | w22]
        where wij in d/2 * d/2 matrix
        set diagonal of wij to 0
        """
        def _clean_diag(w):
            d = w.shape[0]
            if self.original:
                diag_mat = torch.diag(torch.diag(w, 0))
            else:
                if self.disable_block_diag_removal:
                    diag_mat = torch.diag(torch.diag(w, 0))
                else:
                    diag_mat = \
                        torch.diag(torch.diag(w, 0)) + \
                        torch.diag(torch.diag(w, d // 2), d // 2) + \
                        torch.diag(torch.diag(w, -d // 2), d // 2)
            w_res = w - diag_mat
            return w_res
        
        if self.deconv in ['deconv_1', 'deconv_2', None]:
            W = _clean_diag(self.W)
            return W
        elif self.deconv in ['deconv_3', 'deconv_4', 'deconv_4_1', 'deconv_4_2']:
            W1 = _clean_diag(self.W1)
            W2 = _clean_diag(self.W2)
            return W1, W2

    def remove_negative(self, W1, W2):
        W1 = torch.max(W1, self.zero_mat)
        W2 = torch.max(W2, self.zero_mat)
        return W1, W2

    def log_mse_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n, d = target.shape
        loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
        return loss

    def g_dir_loss(self, x):
        if not self.use_g_dir_loss:
            return 0.
        W = self.get_adj()
        return self.log_mse_loss(x @ W, x)

class DagmaTorch:
    """
    Class that implements the DAGMA algorithm
    """
    
    def __init__(self, model: nn.Module, verbose: bool, device : str, dtype: torch.dtype = torch.double):
        """
        Parameters
        ----------
        model : nn.Module
            Neural net that models the structural equations.
        verbose : bool, optional
            If true, the loss/score and h values will print to stdout every ``checkpoint`` iterations,
            as defined in :py:meth:`~dagma.nonlinear.DagmaTorch.fit`. Defaults to ``False``.
        dtype : torch.dtype, optional
            float number precision, by default ``torch.double``.
        """
        self.vprint = print if verbose else lambda *a, **k: None
        self.model : DagmaLinear = model
        self.dtype = dtype
        self.device = device
   

    def minimize(self, 
                 max_iter: float, 
                 lr: float, 
                 lambda1: float, 
                 lambda2: float, 
                 mu: float, 
                 s: float,
                 lr_decay: float = False, 
                 tol: float = 1e-6, 
        ) -> bool:
        r"""
        Solves the optimization problem: 
            .. math::
                \arg\min_{W(\Theta) \in \mathbb{W}^s} \mu \cdot Q(\Theta; \mathbf{X}) + h(W(\Theta)),
        where :math:`Q` is the score function, and :math:`W(\Theta)` is the induced weighted adjacency matrix
        from the model parameters. 
        This problem is solved via (sub)gradient descent using adam acceleration.

        Parameters
        ----------
        max_iter : float
            Maximum number of (sub)gradient iterations.
        lr : float
            Learning rate.
        lambda1 : float
            L1 penalty coefficient. Only applies to the parameters that induce the weighted adjacency matrix.
        lambda2 : float
            L2 penalty coefficient. Applies to all the model parameters.
        mu : float
            Weights the score function.
        s : float
            Controls the domain of M-matrices.
        lr_decay : float, optional
            If ``True``, an exponential decay scheduling is used. By default ``False``.
        tol : float, optional
            Tolerance to admit convergence. Defaults to 1e-6.

        Returns
        -------
        bool
            ``True`` if the optimization succeded. This can be ``False`` when at any iteration, the model's adjacency matrix 
            got outside of the domain of M-matrices.
        """
        self.vprint(f'\nMinimize s={s} -- lr={lr}')
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(.99,.999), weight_decay=mu*lambda2)
        if lr_decay is True:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        obj_prev = 1e16
        for i in range(max_iter):
            X_hat, regressor = self.model(self.X)
            h_val = self.model.h_func(regressor, s)
            if h_val.item() < 0:
                self.vprint(f'Found h negative {h_val.item()} at iter {i}')
                return False
            score = self.model.log_mse_loss(X_hat, self.X)
            g_dir_loss = self.model.g_dir_loss(self.X)
            l1_reg = lambda1 * self.model.l1_reg(regressor)
            obj = mu * (score + l1_reg + g_dir_loss) + h_val
            optimizer.zero_grad()
            obj.backward()
            # print(f"score {score.item():.4f}")
            # print(f"W1 grad {(self.model.W1.grad ** 2).mean().sqrt():.4f}")
            # print(f"W2 grad {(self.model.W2.grad ** 2).mean().sqrt():.4f}")
            optimizer.step()

            if lr_decay and (i+1) % 1000 == 0: #every 1000 iters reduce lr
                scheduler.step()
            if i % self.checkpoint == 0 or i == max_iter-1:
                obj_new = obj.item()
                self.vprint(f"\nInner iteration {i}/{max_iter}")
                self.vprint(f'\th(W(model)): {h_val.item()}')
                self.vprint(f'\tscore(model): {obj_new}')
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    self.vprint(f'\t |(obj_prev({obj_prev:.4f}) - obj_new({obj_new:.4f}) / obj_new({obj_new:.4f})| '
                                f'< tol({tol:.4f})')
                    self.vprint("break")
                    break
                obj_prev = obj_new
        return True

    def fit(self, 
            X: typing.Union[torch.Tensor, np.ndarray],
            lambda1: float = .02, 
            lambda2: float = .005,
            T: int = 4, 
            mu_init: float = .1, 
            mu_factor: float = .1, 
            s: float = 1.0,
            warm_iter: int = 5e4, 
            max_iter: int = 8e4, 
            lr: float = .0002, 
            w_threshold: float = 0.3, 
            checkpoint: int = 1000,
            return_no_filter : bool = False
        ) -> np.ndarray:
        r"""
        Runs the DAGMA algorithm and fits the model to the dataset.

        Parameters
        ----------
        X : typing.Union[torch.Tensor, np.ndarray]
            :math:`(n,d)` dataset.
        lambda1 : float, optional
            Coefficient of the L1 penalty, by default .02.
        lambda2 : float, optional
            Coefficient of the L2 penalty, by default .005.
        T : int, optional
            Number of DAGMA iterations, by default 4.
        mu_init : float, optional
            Initial value of :math:`\mu`, by default 0.1.
        mu_factor : float, optional
            Decay factor for :math:`\mu`, by default .1.
        s : float, optional
            Controls the domain of M-matrices, by default 1.0.
        warm_iter : int, optional
            Number of iterations for :py:meth:`~dagma.nonlinear.DagmaTorch.minimize` for :math:`t < T`, by default 5e4.
        max_iter : int, optional
            Number of iterations for :py:meth:`~dagma.nonlinear.DagmaTorch.minimize` for :math:`t = T`, by default 8e4.
        lr : float, optional
            Learning rate, by default .0002.
        w_threshold : float, optional
            Removes edges with weight value less than the given threshold, by default 0.3.
        checkpoint : int, optional
            If ``verbose`` is ``True``, then prints to stdout every ``checkpoint`` iterations, by default 1000.

        Returns
        -------
        np.ndarray
            Estimated DAG from data.
        
        
        .. important::

            If the output of :py:meth:`~dagma.nonlinear.DagmaTorch.fit` is not a DAG, then the user should try larger values of ``T`` (e.g., 6, 7, or 8) 
            before raising an issue in github.
        """
        torch.set_default_dtype(self.dtype)
        if type(X) == torch.Tensor:
            self.X = X.type(self.dtype)
        elif type(X) == np.ndarray:
            self.X = torch.from_numpy(X).type(self.dtype)
        else:
            ValueError("X should be numpy array or torch Tensor.")
        self.X = self.X.to(self.device)
        
        self.checkpoint = checkpoint
        mu = mu_init
        if type(s) == list:
            if len(s) < T: 
                self.vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.") 
        time0 = time()
        for i in range(int(T)):
            self.vprint(f'\nDagma iter t={i+1} -- mu: {mu}', 30*'-')
            success, s_cur = False, s[i]
            inner_iter = int(max_iter) if i == T - 1 else int(warm_iter)
            model_copy = copy.deepcopy(self.model)
            lr_decay = False
            while success is False:
                success = self.minimize(inner_iter, lr, lambda1, lambda2, mu, s_cur, 
                                    lr_decay)
                if success is False:
                    self.model.load_state_dict(model_copy.state_dict().copy())
                    lr *= 0.5 
                    lr_decay = True
                    if lr < 1e-10:
                        break # lr is too small
                    s_cur = 1
            mu *= mu_factor
            self.vprint(f'\nEnd Dagma iter t={i+1} -- mu: {mu}, time: {time() - time0:.2f}s', 30*'-')
            time0 = time()
        W_est = self.model.get_adj().cpu().detach().numpy()
        W_est_no_filter = W_est.copy()
        W_est[np.abs(W_est) < w_threshold] = 0
        if return_no_filter:
            return W_est_no_filter, W_est
        return W_est    
    
if __name__ == '__main__':
    import utils, os, pickle, utils_dagma
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--d', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:7') 
    args = parser.parse_args()

    
    utils.set_random_seed(args.seed)
    
    n, d = 2000, args.d
    s0 = 4 * d
    version = f"v11/v{d}"
    device = args.device
    data_dir = '/home/jiahang/dagma/src/dagma/simulated_data'
    data_path = os.path.join(data_dir, version, 'X', 'X_1.pkl')

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    X, W_true = data['X'], data['W_true']
    B_true = (W_true != 0)

    eq_model = DagmaLinear(d=d, dagma_type=None, device=device, original=True).to(device)
    model = DagmaTorch(eq_model, device=device, verbose=True)
    print("fit dagma")
    W_est_no_filter, W_est = model.fit(X, lambda1=0.02, lambda2=0.005, return_no_filter=True)
    acc = utils_dagma.count_accuracy(B_true, W_est != 0, use_logger=False)
    print(acc)

    data_path = os.path.join(data_dir, "v39", f"W_{d}_{s0}_{args.seed}.pkl")
    with open(data_path, 'wb') as f:
        pickle.dump(W_est_no_filter, f)
    print("DONE")
