from sklearn.tree import BaseDecisionTree
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from numpy import *
import time
from operator import itemgetter
from multiprocessing import Pool
import numpy as np

from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from cuml.ensemble import RandomForestRegressor as RandomForestRegressor_cuda 
from copy import deepcopy

EARLY_STOP_WINDOW_LENGTH = 25

def compute_feature_importances(estimator):
    if isinstance(estimator, RandomForestRegressor) or isinstance(estimator, ExtraTreesRegressor) :
        return estimator.tree_.compute_feature_importances(normalize=False)
    elif isinstance(estimator, GradientBoostingRegressor):
        n_estimators = len(estimator.estimators_)
        denormalized_importances = estimator.feature_importances_ * n_estimators
        return denormalized_importances
    else:
        importances = [e.tree_.compute_feature_importances(normalize=False)
                       for e in estimator.estimators_]
        importances = array(importances)
        return sum(importances,axis=0) / len(estimator)

class EarlyStopMonitor:

    def __init__(self, window_length=EARLY_STOP_WINDOW_LENGTH):
        """
        :param window_length: length of the window over the out-of-bag errors.
        """

        self.window_length = window_length

    def window_boundaries(self, current_round):
        """
        :param current_round:
        :return: the low and high boundaries of the estimators window to consider.
        """

        lo = max(0, current_round - self.window_length + 1)
        hi = current_round + 1

        return lo, hi

    def __call__(self, current_round, regressor, _):
        """
        Implementation of the GradientBoostingRegressor monitor function API.

        :param current_round: the current boosting round.
        :param regressor: the regressor.
        :param _: ignored.
        :return: True if the regressor should stop early, else False.
        """

        if current_round >= self.window_length - 1:
            lo, hi = self.window_boundaries(current_round)
            return np.mean(regressor.oob_improvement_[lo: hi]) < 0
        else:
            return False

def GENIE3(expr_data,gene_names=None,regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1,use_knockoff=False,use_grnboost2=False):
    
    '''Computation of tree-based scores for all putative regulatory links.
    Notes: GRNBoost2 = GENIE3(RF -> Gradient Boosted Tree) + Early stop + new feature importance measure + SGBM kwargs.
    adopted from arboreto

    Parameters
    ----------
    
    expr_data: numpy array
        Array containing gene expression values. Each row corresponds to a condition and each column corresponds to a gene.
        
    gene_names: list of strings, optional
        List of length p, where p is the number of columns in expr_data, containing the names of the genes. The i-th item of gene_names must correspond to the i-th column of expr_data.
        default: None
        
    regulators: list of strings, optional
        List containing the names of the candidate regulators. When a list of regulators is provided, the names of all the genes must be provided (in gene_names). When regulators is set to 'all', any gene can be a candidate regulator.
        default: 'all'
        
    tree-method: 'RF' or 'ET', optional
        Specifies which tree-based procedure is used: either Random Forest ('RF') or Extra-Trees ('ET')
        default: 'RF'
        
    K: 'sqrt', 'all' or a positive integer, optional
        Specifies the number of selected attributes at each node of one tree: either the square root of the number of candidate regulators ('sqrt'), the total number of candidate regulators ('all'), or any positive integer.
        default: 'sqrt'
         
    ntrees: positive integer, optional
        Specifies the number of trees grown in an ensemble.
        default: 1000
    
    nthreads: positive integer, optional
        Number of threads used for parallel computing
        default: 1
        
        
    Returns
    -------

    An array in which the element (i,j) is the score of the edge directed from the i-th gene to the j-th gene. All diagonal elements are set to zero (auto-regulations are not considered). When a list of candidate regulators is provided, the scores of all the edges directed from a gene that is not a candidate regulator are set to zero.
        
    '''
    
    time_start = time.time()
    
    # Check input arguments
    if not isinstance(expr_data,ndarray):
        raise ValueError('expr_data must be an array in which each row corresponds to a condition/sample and each column corresponds to a gene')
        
    ngenes = expr_data.shape[1]
    target_ngenes = ngenes // 2
    
    if gene_names is not None: # default not in
        if not isinstance(gene_names,(list,tuple)):
            raise ValueError('input argument gene_names must be a list of gene names')
        elif len(gene_names) != ngenes:
            raise ValueError('input argument gene_names must be a list of length p, where p is the number of columns/genes in the expr_data')
        
    if regulators != 'all':
        if not isinstance(regulators,(list,tuple)):
            raise ValueError('input argument regulators must be a list of gene names')

        if gene_names is None: # default not in
            raise ValueError('the gene names must be specified (in input argument gene_names)')
        else:
            sIntersection = set(gene_names).intersection(set(regulators))
            if not sIntersection:
                raise ValueError('the genes must contain at least one candidate regulator')        
        
    if tree_method != 'RF' and tree_method != 'ET':
        raise ValueError('input argument tree_method must be "RF" (Random Forests) or "ET" (Extra-Trees)')
        
    if K != 'sqrt' and K != 'all' and not isinstance(K,int): 
        raise ValueError('input argument K must be "sqrt", "all" or a stricly positive integer')
        
    if isinstance(K,int) and K <= 0:
        raise ValueError('input argument K must be "sqrt", "all" or a stricly positive integer')
    
    if not isinstance(ntrees,int):
        raise ValueError('input argument ntrees must be a stricly positive integer')
    elif ntrees <= 0:
        raise ValueError('input argument ntrees must be a stricly positive integer')
        
    if not isinstance(nthreads,int):
        raise ValueError('input argument nthreads must be a stricly positive integer')
    elif nthreads <= 0:
        raise ValueError('input argument nthreads must be a stricly positive integer')
        
        
    print('Tree method: ' + str(tree_method))
    print('K: ' + str(K))
    print('Number of trees: ' + str(ntrees))
    print('\n')
        
    
    # Get the indices of the candidate regulators
    if regulators == 'all':
        input_idx = list(range(ngenes))
    else:
        input_idx = [i for i, gene in enumerate(gene_names) if gene in regulators]

    
    # Learn an ensemble of trees for each target gene, and compute scores for candidate regulators

    VIM = zeros((ngenes,ngenes))
    
    if nthreads > 1:
        print('running jobs on %d threads' % nthreads)

        input_data = list()
        _ngenes = ngenes if not use_knockoff else target_ngenes
        for i in range(_ngenes):
            _input_idx = deepcopy(input_idx)
            if i in _input_idx:
                _input_idx.remove(i)
            if use_knockoff:
                _input_idx.remove(i + target_ngenes)
            input_data.append( [expr_data,i,_input_idx,tree_method,K,ntrees,use_grnboost2] )

        pool = Pool(nthreads)
        alloutput = pool.map(wr_GENIE3_single, input_data)
    
        for (i,vi) in alloutput:
            VIM[i,:] = vi

    else:
        print('running single threaded jobs')
        _ngenes = ngenes if not use_knockoff else target_ngenes
        for i in range(_ngenes):
            print('Gene %d/%d...' % (i+1,_ngenes))
            _input_idx = deepcopy(input_idx)
            if i in _input_idx:
                _input_idx.remove(i)
            if use_knockoff:
                _input_idx.remove(i + target_ngenes)
            vi = GENIE3_single(expr_data,i,_input_idx,tree_method,K,ntrees,use_grnboost2)
            VIM[i,:] = vi

   
    VIM = transpose(VIM)

    if use_knockoff:
        assert (np.diag(VIM[:target_ngenes, :target_ngenes]) == 0).all()
        assert (np.diag(VIM[target_ngenes:, :target_ngenes]) == 0).all()
 
    time_end = time.time()
    print("Elapsed time: %.2f seconds" % (time_end - time_start))

    return VIM
    
    
    
def wr_GENIE3_single(args):
    return([args[1], GENIE3_single(args[0], args[1], args[2], args[3], args[4], args[5], args[6])])
    


def GENIE3_single(expr_data,output_idx,input_idx,tree_method,K,ntrees,use_grnboost2):
    
    ngenes = expr_data.shape[1]
    
    # Expression of target gene
    output = expr_data[:,output_idx]
    
    # Normalize output data
    output = output / std(output)
    
    # Remove target gene from candidate regulators
    input_idx = input_idx[:]
    if output_idx in input_idx:
        input_idx.remove(output_idx)

    expr_data_input = expr_data[:,input_idx]
    
    # Parameter K of the tree-based method
    if (K == 'all') or (isinstance(K,int) and K >= len(input_idx)):
        max_features = "auto"
    else:
        max_features = K
    if use_grnboost2:
        treeEstimator = GradientBoostingRegressor(
            learning_rate=0.01,
            n_estimators=5000,
            max_features=0.1,
            subsample=0.9
        )
    else:
        if tree_method == 'RF':
            treeEstimator = RandomForestRegressor(n_estimators=ntrees,max_features=max_features)
            
        elif tree_method == 'ET':
            treeEstimator = ExtraTreesRegressor(n_estimators=ntrees,max_features=max_features)

    # Learn ensemble of trees
    if use_grnboost2:
        treeEstimator.fit(expr_data_input,output,monitor=EarlyStopMonitor(EARLY_STOP_WINDOW_LENGTH))
    else:
        treeEstimator.fit(expr_data_input,output)
    
    # Compute importance scores
    feature_importances = compute_feature_importances(treeEstimator)
    vi = zeros(ngenes)
    vi[input_idx] = feature_importances
       
    return vi

if __name__ == '__main__':
    import utils, os, pickle
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--d', type=int, default=10)
    parser.add_argument('--s0', type=int, default=40)
    parser.add_argument('--seed_X', type=int, default=1)
    parser.add_argument('--note', type=str, default="")
    parser.add_argument('--nthreads', type=int, default=1)
    parser.add_argument('--use_grnboost2', action='store_true', default=False) 
    args = parser.parse_args()

    utils.set_random_seed(0)
    
    n, d = 2000, args.d
    s0 = args.s0
    version = f"v11/v{d}_{s0}" + args.note
    root_dir = '/home/jiahang/dagma/src/dagma/simulated_data'
    data_path = os.path.join(root_dir, version, 'X', f'X_{args.seed_X}.pkl')

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    X, W_true = data['X'], data['W_true']
    B_true = (W_true != 0)

    W_est = GENIE3(X, nthreads=args.nthreads, use_grnboost2=args.use_grnboost2)

    prec, rec, threshold = precision_recall_curve(B_true.flatten(), np.abs(W_est).flatten())
    auprc = auc(rec, prec)
    auroc = roc_auc_score(B_true.flatten(), np.abs(W_est).flatten())

    print(f"auprc: {auprc:.2f} | auroc: {auroc:.2f}")

    data_dir = os.path.join(root_dir, "v48", f"{d}_{s0}")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if args.use_grnboost2:
        data_path = os.path.join(data_dir, f"W_{d}_{s0}_{args.seed_X}_0{args.note}_grnboost2.pkl")
    else:
        data_path = os.path.join(data_dir, f"W_{d}_{s0}_{args.seed_X}_0{args.note}.pkl")
    if os.path.exists(data_path):
        raise Exception(f"{data_path} already exists")
    with open(data_path, 'wb') as f:
        pickle.dump(W_est, f)
    print("DONE")


        
        
        