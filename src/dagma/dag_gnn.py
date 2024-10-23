
''''
Main function for traininng DAG-GNN

'''


from __future__ import division
from __future__ import print_function

import argparse
import pickle
import os
import datetime
import time
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import math

import numpy as np

import utils, utils_dagma
from utils_dag_gnn import MLPEncoder, MLPDecoder, encode_onehot, matrix_poly, nll_gaussian, kl_gaussian_sem

parser = argparse.ArgumentParser()

# -----------data parameters ------
# configurations
parser.add_argument('--data_variable_size', type=int, default=10,
                    help='the number of variables in synthetic generated data')

# -----------training hyperparameters
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--suffix', type=str, default='_springs5',
                    help='Suffix for training data (e.g. "_charged".')

args = parser.parse_args()
print(args)

best_ELBO_loss = np.inf
best_NLL_loss = np.inf
best_MSE_loss = np.inf
best_epoch = 0
best_ELBO_graph = []
best_NLL_graph = []
best_MSE_graph = []
# optimizer step on hyparameters
c_A = 1.
lambda_A = 0.
h_A_new = torch.tensor(1.)
h_tol = 1e-8
k_max_iter = 100
h_A_old = np.inf
epochs = 300
batch_size = 100
lr = 3e-3
device = "cuda:7"
n = 2000
d = 60
s0 = 240
data_variable_size = d
graph_type = 'ER'
sem_type = 'gauss'

utils.set_random_seed(0)
torch.set_default_dtype(torch.double)
B_true = utils_dagma.simulate_dag(d, s0, graph_type)
W_true = utils_dagma.simulate_parameter(B_true, [(-1., 1.)])
X = utils_dagma.simulate_linear_sem(W_true, n, sem_type)
X = torch.tensor(X)
X = X.unsqueeze(-1)

dataset = TensorDataset(X, X)
loader = DataLoader(dataset, batch_size=batch_size)
#===================================
# load modules
#===================================
# Generate off-diagonal interaction graph
off_diag = np.ones([data_variable_size, data_variable_size]) - np.eye(data_variable_size)
rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
rel_rec = torch.DoubleTensor(rel_rec)
rel_send = torch.DoubleTensor(rel_send)

# add adjacency matrix A
num_nodes = data_variable_size
adj_A = np.zeros((num_nodes, num_nodes))

encoder = MLPEncoder(1, 64,
                        1, adj_A,
                        batch_size = batch_size,
                        do_prob = 0., factor = True).double()

decoder = MLPDecoder(1, 1, encoder,
                         data_variable_size = data_variable_size,
                         batch_size = batch_size,
                         n_hid=64,
                         do_prob=0.).double()

#===================================
# set up training parameters
#===================================
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=200,
                                gamma=1.)

encoder.to(device)
decoder.to(device)
rel_rec = rel_rec.to(device)
rel_send = rel_send.to(device)


# compute constraint h(A) value
def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

prox_plus = torch.nn.Threshold(0.,0.)

def stau(w, tau):
    w1 = prox_plus(torch.abs(w)-tau)
    return torch.sign(w)*w1


def update_optimizer(optimizer, original_lr, c_A):
    '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
    MAX_LR = 1e-2
    MIN_LR = 1e-4

    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr

    # set LR
    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return optimizer, lr

#===================================
# training:
#===================================

def train(epoch, best_val_loss, B_true, lambda_A, c_A, optimizer):
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []
    shd_train = []

    encoder.train()
    decoder.train()
    for batch_idx, (data, relations) in enumerate(loader):

        
        data, relations = data.to(device), relations.to(device)
        # reshape data
        relations = relations.unsqueeze(2)

        optimizer.zero_grad()

        enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(data, rel_rec, rel_send)  # logits is of size: [num_sims, z_dims]
        edges = logits

        dec_x, output, adj_A_tilt_decoder = decoder(data, edges, data_variable_size * 1, rel_rec, rel_send, origin_A, adj_A_tilt_encoder, Wa)

        if torch.sum(output != output):
            print('nan error\n')

        target = data
        preds = output
        variance = 0.

        # reconstruction accuracy loss
        loss_nll = nll_gaussian(preds, target, variance)

        # KL loss
        loss_kl = kl_gaussian_sem(logits)

        # ELBO loss:
        loss = loss_kl + loss_nll

        # compute h(A)
        h_A = _h_A(origin_A, data_variable_size)
        loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(origin_A*origin_A)


        loss.backward()
        loss = optimizer.step()

        myA.data = stau(myA.data, 0.)

        if torch.sum(origin_A != origin_A):
            print('nan error\n')

        # compute metrics
        graph = origin_A.cpu().detach().numpy()
        graph[np.abs(graph) < 0.3] = 0

        res = utils_dagma.count_accuracy(B_true, graph != 0, use_logger=False)
        fdr, tpr, fpr, shd, nnz = res['fdr'], res['tpr'], res['fpr'], res['shd'], res['nnz']


        mse_train.append(F.mse_loss(preds, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())
        shd_train.append(shd)

    ELBO_loss = np.mean(np.mean(kl_train)  + np.mean(nll_train))
    print(h_A.item())
    print('Epoch: {:04d}'.format(epoch),
          'nll_train: {:.4f}'.format(np.mean(nll_train)),
          'kl_train: {:.4f}'.format(np.mean(kl_train)),
          'ELBO_loss: {:.4f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
          'mse_train: {:.4f}'.format(np.mean(mse_train)),
          'shd_train: {:.4f}'.format(np.mean(shd_train)),
          'time: {:.4f}s'.format(time.time() - t))
    if ELBO_loss < best_val_loss:
        # torch.save(encoder.state_dict(), encoder_file)
        # torch.save(decoder.state_dict(), decoder_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.4f}'.format(np.mean(nll_train)),
              'kl_train: {:.4f}'.format(np.mean(kl_train)),
              'ELBO_loss: {:.4f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
              'mse_train: {:.4f}'.format(np.mean(mse_train)),
              'shd_train: {:.4f}'.format(np.mean(shd_train)),
              'time: {:.4f}s'.format(time.time() - t))


    return ELBO_loss, np.mean(nll_train), np.mean(mse_train), graph, origin_A

#===================================
# main
#===================================
time_st = time.time()
for step_k in range(k_max_iter):
    while c_A < 1e+20:
        for epoch in range(epochs):
            ELBO_loss, NLL_loss, MSE_loss, graph, origin_A = train(epoch, best_ELBO_loss, B_true, lambda_A, c_A, optimizer)
            # update optimizer
            scheduler.step()
            optimizer, lr = update_optimizer(optimizer, lr, c_A)
            if ELBO_loss < best_ELBO_loss:
                best_ELBO_loss = ELBO_loss
                best_epoch = epoch
                best_ELBO_graph = graph

            if NLL_loss < best_NLL_loss:
                best_NLL_loss = NLL_loss
                best_epoch = epoch
                best_NLL_graph = graph

            if MSE_loss < best_MSE_loss:
                best_MSE_loss = MSE_loss
                best_epoch = epoch
                best_MSE_graph = graph

        print("Optimization Finished!")
        print("Best Epoch: {:04d}".format(best_epoch))
        if ELBO_loss > 2 * best_ELBO_loss:
            break

        # update parameters
        A_new = origin_A.data.clone()
        h_A_new = _h_A(A_new, data_variable_size)
        if h_A_new.item() > 0.25 * h_A_old:
            c_A*=10
        else:
            break

        # update parameters
        # h_A, adj_A are computed in loss anyway, so no need to store
    h_A_old = h_A_new.item()
    lambda_A += c_A * h_A_new.item()

    if h_A_new.item() <= h_tol:
        break
print(f"time: {time.time() - time_st:.2f}s")