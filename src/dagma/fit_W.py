import os
import numpy as np
import torch
import torch.optim as optim
from torch import nn
import utils_dagma
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from time import time
from dagma_torch import DagmaLinear, DagmaTorch
from notears_cpu import notears_linear
from golem import golem, postprocess
from dag_gnn import dag_gnn

if __name__ == '__main__':
    import utils, os, pickle, utils_dagma
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--d', type=int, default=None)
    parser.add_argument('--s0', type=int, default=None)
    parser.add_argument('--seed_X', type=int, default=1)
    parser.add_argument('--X_name', type=str, default="")
    parser.add_argument('--dst_version', type=int, default=None)
    parser.add_argument('--dst_name', type=str, default=None)
    parser.add_argument('--model', type=str, required=True, default=['dagma', 'notears', 'golem', 'dag_gnn'])
    parser.add_argument('--device', type=str, default='cuda:7') 

    args = parser.parse_args()

    utils.set_random_seed(0)
    
    n, d = args.n, args.d
    s0 = args.s0
    version = f"v11/v" + args.X_name
    device = args.device
    root_dir = '/home/jiahang/dagma/src/dagma/simulated_data'
    data_path = os.path.join(root_dir, version, 'X', f'X_{args.seed_X}.pkl')

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    X, W_true = data['X'], data['W_true']
    B_true = (W_true != 0)

    print(f"fit {args.model}")
    time_st = time()
    if args.model == 'dagma':
        eq_model = DagmaLinear(d=d, dagma_type='dagma_1', device=device, original=True).to(device)
        model = DagmaTorch(eq_model, device=device, verbose=True)
        W_est_no_filter, W_est = model.fit(X, return_no_filter=True)
    elif args.model == 'notears':
        W_est_no_filter, W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
    elif args.model == 'golem':
        W_est_no_filter = golem(X, lambda_1=2e-2, lambda_2=5.0,
                  equal_variances=True, device=device)
        W_est = postprocess(W_est_no_filter, graph_thres=0.3)
    elif args.model == 'dag_gnn':
        W_est_no_filter, W_est = dag_gnn(X, device)
    print(f"running time: {time() - time_st:.2f}s")

                                    
    acc = utils_dagma.count_accuracy(B_true, W_est != 0, use_logger=False)
    print(acc)

    prec, rec, threshold = precision_recall_curve(B_true.astype(int).flatten(), np.abs(W_est).flatten())
    auprc = auc(rec, prec)
    auroc = roc_auc_score(B_true.astype(int).flatten(), np.abs(W_est).flatten())
    print(f"auprc: {auprc:.2f} | auroc: {auroc:.2f}")

    prec, rec, threshold = precision_recall_curve(B_true.astype(int).flatten(), np.abs(W_est_no_filter).flatten())
    fdp = 1. - prec
    target_fdp_thresh = np.where(fdp[-1:0:-1] < 0.2)[0][-1]
    power = rec[-1:0:-1][target_fdp_thresh]

    print(f"Given fdp < 0.2, the maximal power is {power}")

    if args.dst_version is not None and args.dst_name is not None:
        data_dir = os.path.join(root_dir, f"v{args.dst_version}", f"{n}_{d}_{s0}")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        data_path = os.path.join(data_dir, f"{args.dst_name}.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump(W_est_no_filter, f)
    else:
        print("No dst_version or dst_name, not saving data")
    print("DONE")

