import math
import numpy as np
import utils

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.optim import Adam, lr_scheduler


class Deconv(nn.Module):
    def __init__(self, configs: dict) -> None:
        super().__init__()

        """
        TEST
        """
        self.device = configs['device']
        # self.device = 'cpu'
        self.order = configs['order']
        # self.order = 5
        self.d = configs['d'] * 2
        # self.d = 120
        self.clean_diag_tag = configs['clean_diag']
        # self.clean_diag_tag = True

        self.W = Parameter(
            torch.empty((self.d, self.d), device=self.device)
        )
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.W = self.clean_diag(self.W)

    def clean_diag(self, W):
        """
        w is of shape [d/2 + d/2, d/2 + d/2], that is, w is
            [w11 | w12]
            [----|----]
            [w21 | w22]
        where wij in d/2 * d/2 matrix
        set diagonal of wij to 0
        """
        if self.clean_diag_tag:
            with torch.no_grad():
                W[np.eye(self.d).astype(bool)] = 0.
                W[np.eye(self.d, k = self.d // 2).astype(bool)] = 0.
                W[np.eye(self.d, k = - self.d // 2).astype(bool)] = 0.
        return W

    def forward(self):
        # TODO: accumulated operations could have numeric problems
        ## consider 1) initialization, 2) (layer) normalization
        W_cum = []
        self.W = self.clean_diag(self.W)
        W_cum.append(self.W)
        for i in range(1, self.order):
            W_cum.append(W_cum[i-1] @ self.W)
            W_cum[-1] = self.clean_diag(W_cum[-1])
        W_cum = torch.stack(W_cum)
        W = W_cum.sum(dim=0)
        W = self.clean_diag(W)
        return W
    
    def get_W_dir(self):
        W = self.clean_diag(self.W)
        return W
    
def net_deconv(W_est: np.ndarray, configs: dict):
    # TODO: remove diag of returned W_pred
    # TODO: rescaling W_est to make W_pred converge
    # TODO: rescaling W_pred to make itself converge
    # TODO: add dag constraints to push the W_pred to be a DAG
    # TODO: l2 regularizer
    # TODO: l1 regularizer

    """
    TEST
    """
    device = configs['device']
    # device = 'cpu'
    epochs = configs['epochs']
    # epochs = int(3e4)
    lr = configs['lr']
    # lr = 1e-2
    l2_w = configs['l2_w']
    # l2_w = 0.
    dag_control = configs['dag_control_deconv']

    if dag_control == 'dag_1':
        W = np.abs(W_est.copy())
        mask = utils.extract_dag_mask(W, 0)
        W_est[~mask] = 0.

    W_est = torch.tensor(W_est, device=device, dtype=torch.float)
    model = Deconv(configs)
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=l2_w)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    for e in range(epochs):
        W_pred = model()
        optimizer.zero_grad()
        loss = loss_fn(W_pred, W_est)
        loss.backward()
        optimizer.step()

        if e % 200 == 0:
            print(f"Epoch {e} | Loss {loss.item():.6f}")

        if e == 1000:
            scheduler.step()

    return model.get_W_dir().detach().cpu().numpy()
    
if __name__ == '__main__':
    import pickle

    with open('/Users/jiahang/Documents/dagma/src/dagma/simulated_data/v11/v60/W/W_6_0.pkl', 'rb') as f:
        W = pickle.load(f)
    W_est = W['W_est']
    W_dir = net_deconv(W_est, {})
    pass