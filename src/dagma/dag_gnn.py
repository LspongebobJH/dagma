
''''
Main function for traininng DAG-GNN

'''


from __future__ import division
from __future__ import print_function


import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import math

import numpy as np
import time

from utils_dag_gnn import MLPEncoder, MLPDecoder, encode_onehot, matrix_poly, nll_gaussian, kl_gaussian_sem

# compute constraint h(A) value
def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

def stau(w, tau):
    prox_plus = torch.nn.Threshold(0.,0.)
    w1 = prox_plus(torch.abs(w)-tau)
    return torch.sign(w)*w1


def update_optimizer(optimizer, original_lr, c_dag):
    '''related LR to c_dag, whenever c_dag gets big, reduce LR proportionally'''
    MAX_LR = 1e-2
    MIN_LR = 1e-4

    estimated_lr = original_lr / (math.log10(c_dag) + 1e-10)
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

def train(epoch, lambda_l1, lambda_dag, c_dag, optimizer,
        encoder, decoder, loader, device, data_variable_size,
        rel_rec, rel_send):
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []
    all_loss_train = []
    

    encoder.train()
    decoder.train()
    for batch_idx, (data, relations) in enumerate(loader):

        
        data, relations = data.to(device), relations.to(device)
        # reshape data
        relations = relations.unsqueeze(2)

        optimizer.zero_grad()

        l1_loss, enc_x, logits, W_est_no_filter, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(data, rel_rec, rel_send)  # logits is of size: [num_sims, z_dims]
        edges = logits

        l1_loss, dec_x, output, adj_A_tilt_decoder = decoder(data, edges, data_variable_size * 1, rel_rec, rel_send, W_est_no_filter, adj_A_tilt_encoder, Wa)

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
        h_A = _h_A(W_est_no_filter, data_variable_size)
        loss += lambda_dag * h_A + 0.5 * c_dag * h_A * h_A + 100. * torch.trace(W_est_no_filter*W_est_no_filter)
        loss += l1_loss * lambda_l1


        loss.backward()
        optimizer.step()

        myA.data = stau(myA.data, 0.)

        if torch.sum(W_est_no_filter != W_est_no_filter):
            print('nan error\n')

        # compute metrics
        W_est = W_est_no_filter.cpu().detach().numpy()
        W_est[np.abs(W_est) < 0.3] = 0

        mse_train.append(F.mse_loss(preds, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())
        all_loss_train.append(loss.item())

    # print('Epoch: {:04d}'.format(epoch),
    #       'nll_train: {:.10f}'.format(np.mean(nll_train)),
    #       'kl_train: {:.10f}'.format(np.mean(kl_train)),
    #       'ELBO_loss: {:.10f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
    #       'mse_train: {:.10f}'.format(np.mean(mse_train)),
    #       'time: {:.4f}s'.format(time.time() - t))

    ELBO_loss = np.mean(np.mean(kl_train)  + np.mean(nll_train))
    all_loss = np.mean(all_loss_train)

    print('Epoch: {:04d}'.format(epoch),
          'tot loss: {:.6f}'.format(all_loss),
          'h(A): {:.6f}'.format(h_A.item()),
          'time: {:.2f}s'.format(time.time() - t))

    # return ELBO_loss, np.mean(nll_train), np.mean(mse_train), graph, origin_A
    return all_loss, ELBO_loss, np.mean(nll_train), np.mean(mse_train), W_est, W_est_no_filter

#===================================
# main
#===================================
def dag_gnn(X, lambda_l1, lambda_l2, device):
    n = X.shape[0]
    d = X.shape[1]

    best_epoch = 0
    # optimizer step on hyparameters
    c_dag = 1.
    lambda_dag = 0.
    h_A_new = torch.tensor(1.)
    # h_tol = 1e-8
    h_tol = 1e-6
    k_max_iter = 100
    h_A_old = np.inf
    epochs = 300
    batch_size = n
    lr = 3e-3
    data_variable_size = d

    # B_true = utils_dagma.simulate_dag(d, s0, graph_type)
    # W_true = utils_dagma.simulate_parameter(B_true, [(-1., 1.)])
    # X = utils_dagma.simulate_linear_sem(W_true, n, sem_type)
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
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr=lr, weight_decay=lambda_l2)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200,
                                    gamma=1.)

    encoder.to(device)
    decoder.to(device)
    rel_rec = rel_rec.to(device)
    rel_send = rel_send.to(device)

    tot_best_all_loss = np.inf
    W_est_no_filter_best = None

    for step_k in range(k_max_iter):
        while c_dag < 1e+20:
            all_loss_min = np.inf
            cnt = 0
            for epoch in range(epochs):
                all_loss, ELBO_loss, NLL_loss, MSE_loss, W_est, W_est_no_filter = train(
                    epoch, lambda_l1, lambda_dag, c_dag, optimizer, 
                    encoder, decoder, loader, device, data_variable_size,
                    rel_rec, rel_send)
                # update optimizer
                scheduler.step()
                optimizer, lr = update_optimizer(optimizer, lr, c_dag)
                    
                if all_loss < all_loss_min - 1e-6:
                    best_epoch = epoch
                    all_loss_min = all_loss
                    cnt = 0
                    W_est_no_filter_best = W_est_no_filter.clone()

                else:
                    cnt += 1

                if cnt >= 10:
                    break


            print("Optimization Finished!")
            print("Best Epoch: {:04d}".format(best_epoch))
            if all_loss_min > 2 * tot_best_all_loss:
                break
            else:
                tot_best_all_loss = all_loss_min

            # update parameters
            A_new = W_est_no_filter_best.clone()
            h_A_new = _h_A(A_new, data_variable_size)
            if h_A_new.item() > 0.25 * h_A_old:
                c_dag*=10
            else:
                break

            # update parameters
            # h_A, adj_A are computed in loss anyway, so no need to store
        h_A_old = h_A_new.item()
        lambda_dag += c_dag * h_A_new.item()

        if h_A_new.item() <= h_tol:
            break

    return W_est_no_filter_best.cpu().detach().numpy(), W_est # W_est_no_filter, W_est

# if __name__ == '__main__':
    