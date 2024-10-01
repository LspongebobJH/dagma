from sklearn.tree import BaseDecisionTree
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from numpy import *
import time
from operator import itemgetter
from multiprocessing import Pool
import numpy as np

from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.inspection import permutation_importance
from copy import deepcopy

EARLY_STOP_WINDOW_LENGTH = 25

def compute_feature_importances(estimator, importance='original', X=None, y=None):
    if importance == 'original':
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

    elif importance == 'permutation':
        return permutation_importance(estimator, X, y)['importances_mean']
    
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

def preprocess_input_idx_expr_data(expr_data, knock_genie3_type, input_idx: list, use_knockoff: bool, remove_self: bool, 
                    i: int, target_ngenes: int, input_ngenes: int):
    _input_idx = deepcopy(input_idx)
    if remove_self:
        # input_idx
        if i in _input_idx:
            _input_idx.remove(i)
        if use_knockoff:
            _input_idx.remove(i + target_ngenes)
        
    if use_knockoff:
        if knock_genie3_type == 'separate':
            _expr_data = np.zeros((expr_data['X'].shape[0], input_ngenes))
            _expr_data[:, :target_ngenes] = expr_data['X']
            _expr_data[:, target_ngenes+1:] = expr_data['X_tilde'][i] 
            # we manipulate _expr_data since they are insufficient elements in expr_data['X_tilde'][i]
            # to fill in _expr_data[:, target_ngenes:]
            
        elif knock_genie3_type == 'unified':
            _expr_data = expr_data

    else:
        _expr_data = expr_data
    
    return _input_idx, _expr_data


def GENIE3(expr_data, 
           importance='original',
           knock_genie3_type=None,
           gene_names=None,
           regulators='all',
           tree_method='RF',
           K='sqrt',
           ntrees=1000,
           nthreads=1,
           use_knockoff=False,
           use_grnboost2=False,
           disable_remove_self=False,
           disable_norm=False,
           tune_params=None,
           model_type='tree'
           ):
    
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
    if gene_names is not None or regulators != 'all':
        assert use_knockoff == False, "use_knockoff is not supported for bipartite graph yet"
    time_start = time.time()
    if use_knockoff:
        assert knock_genie3_type is not None and knock_genie3_type in ['separate', 'unified']

    # Check input arguments
    if not use_knockoff and not isinstance(expr_data,ndarray):
        raise ValueError('expr_data must be an array in which each row corresponds to a condition/sample and each column corresponds to a gene')
    
    if use_knockoff:
        if knock_genie3_type == 'separate':
            target_ngenes = expr_data['X'].shape[1]
            all_ngenes = expr_data['X'].shape[1] * 2
        elif knock_genie3_type == 'unified':
            target_ngenes = expr_data.shape[1] // 2
            all_ngenes = expr_data.shape[1]
    else:
        # TODO: bipartite graph don't remove TF from target genes for now
        # because there could be TF as target genes in real data
        # though in this simulation, no.
        target_ngenes = expr_data.shape[1] 
        all_ngenes = target_ngenes
    
    if gene_names is not None: # default not in
        if not isinstance(gene_names,(list,tuple)):
            raise ValueError('input argument gene_names must be a list of gene names')
        elif len(gene_names) != target_ngenes:
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
        input_idx = list(range(all_ngenes))
    else:
        input_idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
    input_ngenes = len(input_idx)

    
    # Learn an ensemble of trees for each target gene, and compute scores for candidate regulators

    # actual only the [:, :target_ngenes] being used.
    VIM = zeros((all_ngenes,all_ngenes))
    
    if nthreads > 1:
        print('running jobs on %d threads' % nthreads)

        input_data = list()
        for i in range(target_ngenes):
            _input_idx, _expr_data = \
                preprocess_input_idx_expr_data(expr_data, knock_genie3_type, input_idx, use_knockoff, 
                                               not disable_remove_self, 
                                               i, target_ngenes, input_ngenes)
            input_data.append( [_expr_data,i,_input_idx,tree_method,K,ntrees,use_grnboost2,disable_norm,tune_params,importance,model_type] )

        pool = Pool(nthreads)
        alloutput = pool.map(wr_GENIE3_single, input_data)
    
        for (i,vi) in alloutput:
            VIM[i,:] = vi

    else:
        print('running single threaded jobs')
        for i in range(target_ngenes):
            print('Gene %d/%d...' % (i+1,target_ngenes))
            _input_idx, _expr_data = \
                preprocess_input_idx_expr_data(expr_data, knock_genie3_type, input_idx, use_knockoff, 
                                               not disable_remove_self, 
                                               i, target_ngenes, input_ngenes)
            vi = GENIE3_single(_expr_data,i,_input_idx,tree_method,K,ntrees,use_grnboost2,disable_norm,tune_params,importance,model_type)
            VIM[i,:] = vi

   
    VIM = transpose(VIM)

    if use_knockoff and not disable_remove_self:
        assert (np.diag(VIM[:target_ngenes, :target_ngenes]) == 0).all()
        assert (np.diag(VIM[target_ngenes:, :target_ngenes]) == 0).all()
 
    time_end = time.time()
    print("Elapsed time: %.2f seconds" % (time_end - time_start))

    return VIM
    
    
    
def wr_GENIE3_single(args):
    return([args[1], GENIE3_single(*args)])
    


def GENIE3_single(expr_data,output_idx,input_idx,tree_method,K,ntrees,use_grnboost2,disable_norm,tune_params,importance,
                  model_type):
    
    ngenes = expr_data.shape[1]
    
    # Expression of target gene
    output = expr_data[:,output_idx]
    
    # Normalize output data
    if not disable_norm:
        output = output / std(output)
    
    # Remove target gene from candidate regulators
    # input_idx = input_idx[:]
    # if output_idx in input_idx:
    #     input_idx.remove(output_idx)

    expr_data_input = expr_data[:,input_idx]
    
    # Parameter K of the tree-based method
    if (K == 'all') or (isinstance(K,int) and K >= len(input_idx)):
        max_features = "auto"
    else:
        max_features = K
    if model_type == 'tree':
        if use_grnboost2:
            n_estimators = tune_params['ntrees'] if tune_params['ntrees'] is not None else 5000
            max_features = tune_params['max_feat'] if tune_params['ntrees'] is not None else 0.1
            subsample = tune_params['max_sample'] if tune_params['ntrees'] is not None else 0.9
            treeEstimator = GradientBoostingRegressor(
                learning_rate=0.01,
                n_estimators=n_estimators,
                max_features=max_features,
                subsample=subsample
            )
        else:
            n_estimators = tune_params['ntrees'] if tune_params['ntrees'] is not None else 1000
            max_features = tune_params['max_feat'] if tune_params['ntrees'] is not None else 'sqrt'
            max_samples = tune_params['max_sample'] if tune_params['ntrees'] is not None else None
            if tree_method == 'RF':
                treeEstimator = RandomForestRegressor(n_estimators=n_estimators,
                                                    max_features=max_features,
                                                    max_samples=max_samples)
                
            elif tree_method == 'ET':
                treeEstimator = ExtraTreesRegressor(n_estimators=n_estimators,
                                                    max_features=max_features,
                                                    max_samples=max_samples)
    

        # Learn ensemble of trees
        if use_grnboost2:
            treeEstimator.fit(expr_data_input,output,monitor=EarlyStopMonitor(EARLY_STOP_WINDOW_LENGTH))
        else:
            treeEstimator.fit(expr_data_input,output)
        
        # Compute importance scores
        feature_importances = compute_feature_importances(treeEstimator, 
                                                        importance, 
                                                        expr_data_input,
                                                        output)
    elif model_type in ['OLS', 'L1', 'L2', 'L1+L2']:
        if model_type == 'OLS':
            from sklearn.linear_model import LinearRegression
            clf = LinearRegression()

        elif model_type == 'L1':
            from sklearn.linear_model import Lasso
            clf = Lasso()

        elif model_type == 'L2':
            from sklearn.linear_model import Ridge
            clf = Ridge()

        elif model_type == 'L1+L2': 
            from sklearn.linear_model import ElasticNet
            clf = ElasticNet()

        clf.fit(expr_data_input, output)
        feature_importances = clf.coef_

    vi = zeros(ngenes)
    vi[input_idx] = feature_importances
       
    return vi

if __name__ == '__main__':
    import utils, os, pickle
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--d', type=int, default=100)
    parser.add_argument('--d1', type=int, default=None)
    parser.add_argument('--d2', type=int, default=None)
    parser.add_argument('--s0', type=int, default=600)
    parser.add_argument('--seed_X', type=int, default=2)
    parser.add_argument('--src_note', type=str, default="")
    parser.add_argument('--dst_note', type=str, default="")
    parser.add_argument('--nthreads', type=int, default=1)
    parser.add_argument('--model_type', type=str, default="tree")
    parser.add_argument('--use_grnboost2', action='store_true', default=False)
    parser.add_argument('--disable_norm', action='store_true', default=False) 
    parser.add_argument('--force_save', action='store_true', default=False) 

    # tune hyperparameters of tree models
    parser.add_argument('--ntrees', type=int, default=None)
    parser.add_argument('--max_feat', type=float, default=None)
    parser.add_argument('--max_sample', type=float, default=None)
    parser.add_argument('--importance', type=str, default='original', choices=['original', 'permutation'])
    
    args = parser.parse_args()

    assert args.d1 is None and args.d2 is None
    utils.set_random_seed(0)
    
    n, d = 2000, args.d
    d1, d2 = args.d1, args.d2
    s0 = args.s0
    version = f"v11/v{d}_{s0}" + args.src_note
    root_dir = '/home/jiahang/dagma/src/dagma/simulated_data'
    # root_dir = '/Users/jiahang/Documents/dagma/src/dagma/simulated_data'
    tune_params = {
        'ntrees': args.ntrees,
        'max_feat': args.max_feat,
        'max_sample': args.max_feat,
    }

    data_path = os.path.join(root_dir, version, 'X', f'X_{args.seed_X}.pkl')

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    X, W_true = data['X'], data['W_true']
    B_true = (W_true != 0)


    # X = np.random.normal(size=(2000, 100))
    # B_true = np.ones((100, 100))
    if d1 is None and d2 is None:
        W_est = GENIE3(X, 
                       nthreads=args.nthreads, 
                       use_grnboost2=args.use_grnboost2, 
                       disable_norm=args.disable_norm,
                       tune_params=tune_params,
                       importance=args.importance,
                       model_type=args.model_type)
    else:
        W_est = GENIE3(X, 
                       gene_names=list(range(d)),
                       regulators=list(range(d1)),
                       nthreads=args.nthreads, 
                       use_grnboost2=args.use_grnboost2, 
                       disable_norm=args.disable_norm,
                       model_type=args.model_type)

    prec, rec, threshold = precision_recall_curve(B_true.astype(int).flatten(), np.abs(W_est).flatten())
    auprc = auc(rec, prec)
    auroc = roc_auc_score(B_true.astype(int).flatten(), np.abs(W_est).flatten())

    prec_trunc, rec_trunc, threshold_trunc = \
        precision_recall_curve(B_true[:d1, :].astype(int).flatten(), 
                               np.abs(W_est[:d1, :]).flatten())
    auprc_trunc = auc(rec_trunc, prec_trunc)
    auroc_trunc = roc_auc_score(B_true[:d1, :].astype(int).flatten(), 
                                np.abs(W_est[:d1, :]).flatten())

    print(f"auprc: {auprc:.2f} | auroc: {auroc:.2f}")
    print(f"auprc_trunc: {auprc_trunc:.2f} | auroc_trunc: {auroc_trunc:.2f}")

    data_dir = os.path.join(root_dir, "v48", f"{d}_{s0}")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_path = os.path.join(data_dir, f"W_{d}_{s0}_{args.seed_X}_0{args.dst_note}.pkl")
    if os.path.exists(data_path) and not args.force_save:
        raise Exception(f"{data_path} already exists")
    with open(data_path, 'wb') as f:
        pickle.dump(W_est, f)
    print("DONE")


        
        
        