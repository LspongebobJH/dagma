from locally_connected import LocallyConnected
import torch
import torch.nn as nn
import numpy as np
from  torch import optim
import copy
from tqdm.auto import tqdm
import typing
import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.optim import Adam, lr_scheduler

__all__ = ["DagmaMLP", "DagmaTorch"]

# TODO: rescaling W_est to make W_pred converge
# TODO: rescaling W_pred to make itself converge
# TODO: add dag constraints to push the W_pred to be a DAG
# TODO: l2 regularizer
# TODO: l1 regularizer

class DagmaMLP(nn.Module): 
    """
    Class that models the structural equations for the causal graph using MLPs.
    """
    
    def __init__(self, dims: typing.List[int], bias: bool = True, dtype: torch.dtype = torch.double, 
                 device : str = 'cuda:7'):
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
        self.device = device
        super(DagmaMLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        self.dims, self.d = dims, dims[0]
        self.I = torch.eye(self.d).to(self.device)
        self.fc1 = nn.Linear(self.d, self.d * dims[1], bias=bias)
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(self.d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n, d] -> [n, d]
        r"""
        Applies the current states of the structural equations to the dataset X

        Parameters
        ----------
        x : torch.Tensor
            Input dataset with shape :math:`(n,d)`.

        Returns
        -------
        torch.Tensor
            Result of applying the structural equations to the input data.
            Shape :math:`(n,d)`.
        """
        x = self.fc1(x)
        x = x.view(-1, self.dims[0], self.dims[1])
        for fc in self.fc2:
            x = torch.sigmoid(x)
            x = fc(x)
        x = x.squeeze(dim=2)
        return x

    def h_func(self, s: float = 1.0) -> torch.Tensor:
        r"""
        Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG

        Parameters
        ----------
        s : float, optional
            Controls the domain of M-matrices, by default 1.0

        Returns
        -------
        torch.Tensor
            A scalar value of the log-det acyclicity function :math:`h(\Theta)`.
        """
        fc1_weight = self.fc1.weight
        fc1_weight = fc1_weight.view(self.d, -1, self.d)
        A = torch.sum(fc1_weight ** 2, dim=1).t()  # [i, j]
        h = -torch.slogdet(s * self.I - A)[1] + self.d * np.log(s)
        return h

    def fc1_l1_reg(self) -> torch.Tensor:
        r"""
        Takes L1 norm of the weights in the first fully-connected layer

        Returns
        -------
        torch.Tensor
            A scalar value of the L1 norm of first FC layer. 
        """
        return torch.sum(torch.abs(self.fc1.weight))

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        r"""
        Computes the induced weighted adjacency matrix W from the first FC weights.
        Intuitively each edge weight :math:`(i,j)` is the *L2 norm of the functional influence of variable i to variable j*.

        Returns
        -------
        np.ndarray
            :math:`(d,d)` weighted adjacency matrix 
        """
        fc1_weight = self.fc1.weight
        fc1_weight = fc1_weight.view(self.d, -1, self.d)  
        A = torch.sum(fc1_weight ** 2, dim=1).t() 
        W = torch.sqrt(A)
        W = W.cpu().detach().numpy()  # [i, j]
        return W

class DagmaLinear(nn.Module): 
    """
    Class that models the structural equations for the linear causal graph.
    """
    
    def __init__(self, d : int, dtype: torch.dtype = torch.double, 
                 deconv_type_dagma: str = "deconv_1",
                 ord_dagma: int = 5,
                 device : str = 'cuda:7'):
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
        self.d = 2 * d
        self.device = device
        self.deconv = deconv_type_dagma
        self.order = ord_dagma
        super(DagmaLinear, self).__init__()
        self.I = torch.eye(self.d).to(self.device)
        if self.deconv in ['deconv_1', 'deconv_2']:
            self.W = Parameter(
                torch.empty((self.d, self.d), device=self.device)
            )
            init.kaiming_uniform_(self.W, a=math.sqrt(5))
        else:
            self.fc1 = nn.Linear(self.d, self.d, bias=False)
            nn.init.zeros_(self.fc1.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n, d] -> [n, d]
        r"""
        Applies the current states of the structural equations to the dataset X

        Parameters
        ----------
        x : torch.Tensor
            Input dataset with shape :math:`(n,d)`.

        Returns
        -------
        torch.Tensor
            Result of applying the structural equations to the input data.
            Shape :math:`(n,d)`.
        """
        if self.deconv == 'deconv_1':
            W_cum = []
            W_cum.append(self.W)
            for i in range(1, self.order):
                W_cum.append(W_cum[i-1] @ self.W)
            W_cum = torch.stack(W_cum)
            W = W_cum.sum(dim=0)
            x = x @ W
        elif self.deconv == 'deconv_2':
            pass
        else:
            x = self.fc1(x)
        return x

    def h_func(self, s: float = 1.0) -> torch.Tensor:
        r"""
        Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG

        Parameters
        ----------
        s : float, optional
            Controls the domain of M-matrices, by default 1.0

        Returns
        -------
        torch.Tensor
            A scalar value of the log-det acyclicity function :math:`h(\Theta)`.
        """
        fc1_weight = self.get_W()
        A = fc1_weight ** 2  # [i, j]
        h = -torch.slogdet(s * self.I - A)[1] + self.d * np.log(s)
        return h

    def fc1_l1_reg(self) -> torch.Tensor:
        r"""
        Takes L1 norm of the weights in the first fully-connected layer

        Returns
        -------
        torch.Tensor
            A scalar value of the L1 norm of first FC layer. 
        """
        w = self.get_W()
        return torch.sum(torch.abs(w))

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        r"""
        Computes the induced weighted adjacency matrix W from the first FC weights.
        Intuitively each edge weight :math:`(i,j)` is the *L2 norm of the functional influence of variable i to variable j*.

        Returns
        -------
        np.ndarray
            :math:`(d,d)` weighted adjacency matrix 
        """
        return self.get_W().cpu().detach().numpy()

    def get_W(self):
        if self.deconv == 'deconv_1':
            return self.W
        else:
            return self.fc1.weight.T

class DagmaTorch:
    """
    Class that implements the DAGMA algorithm
    """
    
    def __init__(self, model: nn.Module, verbose: bool = False, dtype: torch.dtype = torch.double, 
                 deconv_type_dagma: str = "deconv_1",
                 device : str = 'cuda:7', dagma_type : str = None):
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
        self.model = model
        self.dtype = dtype
        self.device = device
        self.dagma_type = dagma_type
        self.deconv_type_dagma = deconv_type_dagma
    
    def log_mse_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the logarithm of the MSE loss:
            .. math::
                \frac{d}{2} \log\left( \frac{1}{n} \sum_{i=1}^n (\mathrm{output}_i - \mathrm{target}_i)^2 \right)
        
        Parameters
        ----------
        output : torch.Tensor
            :math:`(n,d)` output of the model
        target : torch.Tensor
            :math:`(n,d)` input dataset

        Returns
        -------
        torch.Tensor
            A scalar value of the loss.
        """
        n, d = target.shape
        loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
        return loss

    def clean_diag(self):
        """
        w is of shape [d/2 + d/2, d/2 + d/2], that is, w is
            [w11 | w12]
            [----|----]
            [w21 | w22]
        where wij in d/2 * d/2 matrix
        set diagonal of wij to 0
        """
        with torch.no_grad():
            w = self.model.get_W()
            d = w.shape[0]
            w[np.eye(d).astype(bool)] = 0.
            w[np.eye(d, k = d // 2).astype(bool)] = 0.
            w[np.eye(d, k = - d // 2).astype(bool)] = 0.

    def check_diag(self):
        """
        w is of shape [d/2 + d/2, d/2 + d/2], that is, w is
            [w11 | w12]
            [----|----]
            [w21 | w22]
        where wij in d/2 * d/2 matrix
        ckech whether diagonal of wij to 0
        """
        w = self.model.get_W()
        d = w.shape[0]
        if (w[np.eye(d).astype(bool)] != 0.).any() or \
            (w[np.eye(d, k = d // 2).astype(bool)] != 0.).any() or \
            (w[np.eye(d, k = - d // 2).astype(bool)] != 0.).any():
                return False
        return True

    def minimize(self, 
                 max_iter: float, 
                 lr: float, 
                 lambda1: float, 
                 lambda2: float, 
                 mu: float, 
                 s: float,
                 lr_decay: float = False, 
                 tol: float = 1e-6, 
                 pbar: typing.Optional[tqdm] = None,
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
        pbar : tqdm, optional
            Controls bar progress. Defaults to ``tqdm()``.

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
            optimizer.zero_grad()
            h_val = self.model.h_func(s)
            if h_val.item() < 0:
                self.vprint(f'Found h negative {h_val.item()} at iter {i}')
                return False
            X_hat = self.model(self.X)
            score = self.log_mse_loss(X_hat, self.X)
            l1_reg = lambda1 * self.model.fc1_l1_reg()
            obj = mu * (score + l1_reg) + h_val
            obj.backward()
            optimizer.step()

            if self.dagma_type == 'dagma_1':
                self.clean_diag()
                assert self.check_diag()

            if lr_decay and (i+1) % 1000 == 0: #every 1000 iters reduce lr
                scheduler.step()
            if i % self.checkpoint == 0 or i == max_iter-1:
                obj_new = obj.item()
                self.vprint(f"\nInner iteration {i}")
                self.vprint(f'\th(W(model)): {h_val.item()}')
                self.vprint(f'\tscore(model): {obj_new}')
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    pbar.update(max_iter-i)
                    break
                obj_prev = obj_new
            pbar.update(1)
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
        with tqdm(total=(T-1)*warm_iter+max_iter) as pbar:
            for i in range(int(T)):
                self.vprint(f'\nDagma iter t={i+1} -- mu: {mu}', 30*'-')
                success, s_cur = False, s[i]
                inner_iter = int(max_iter) if i == T - 1 else int(warm_iter)
                model_copy = copy.deepcopy(self.model)
                lr_decay = False
                while success is False:
                    success = self.minimize(inner_iter, lr, lambda1, lambda2, mu, s_cur, 
                                        lr_decay, pbar=pbar)
                    if self.dagma_type == 'dagma_1':
                        self.clean_diag()
                        assert self.check_diag()
                    if success is False:
                        self.model.load_state_dict(model_copy.state_dict().copy())
                        lr *= 0.5 
                        lr_decay = True
                        if lr < 1e-10:
                            break # lr is too small
                        s_cur = 1
                mu *= mu_factor
        if self.dagma_type == 'dagma_1':
            assert self.check_diag()
        W_est = self.model.fc1_to_adj()
        W_est_no_filter = W_est.copy()
        W_est[np.abs(W_est) < w_threshold] = 0
        if return_no_filter:
            return W_est_no_filter, W_est
        return W_est
        


def test():
    import utils_dagma as utils
    
    utils.set_random_seed(1)
    torch.manual_seed(1)
    
    n, d, s0, graph_type, sem_type = 1000, 20, 50, 'ER', 'gauss'
    device = 'cuda:7'
    print("simulated dag")
    B_true = utils.simulate_dag(d, s0, graph_type)
    # print("simulated nonlinear SEM")
    # X = utils.simulate_nonlinear_sem(B_true, n, sem_type)
    print("simulated linear dag")
    X = utils.simulate_linear_sem(B_true, n, sem_type)

    eq_model = DagmaLinear(d=d, device=device).to(device)
    model = DagmaTorch(eq_model, device=device, verbose=True, dagma_type='dagma_1')
    print("fit dagma")
    W_est = model.fit(X, lambda1=0.02, lambda2=0.005)
    acc = utils.count_accuracy(B_true, W_est != 0, use_logger=False)
    print(acc)
    
    
if __name__ == '__main__':
    test()
