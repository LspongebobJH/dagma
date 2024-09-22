import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from hidimstat.knockoffs.gaussian_knockoff import (_estimate_distribution,
                                                   gaussian_knockoff_generation)

from sklearn.covariance import (GraphicalLassoCV, empirical_covariance,
                                ledoit_wolf)
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter
from sklearn.linear_model import (Lasso,
                                  LogisticRegressionCV,
                                  ElasticNet)
from sklearn.cross_decomposition import PLSRegression

import utils
import xgboost as xgb
from cuml import LinearRegression
import cupy as cp

#########################
# Main 
#########################

def get_knockoffs_stats(X, configs, n_jobs=1,
                        gaussian=False,
                        cov_estimator='graph_lasso'):
    """
    X: n * p, feature matrix
    gaussian: if True, use second-order knockoff for Gaussian cases from candes paper
    cov_estimator is used by "gaussian'
    """
    method_diagn_gen = configs['method_diagn_gen']
    lasso_alpha = configs['lasso_alpha']
    PLS_n_comp = configs['PLS_n_comp']
    device = configs['device']
    
    if gaussian: # second-order knockoff for Gaussian cases, candes paper, baselines
        mu, Sigma = _estimate_distribution(
        X, shrink=True, cov_estimator=cov_estimator, n_jobs=n_jobs)
        
        X_tildes = gaussian_knockoff_generation(X, mu, Sigma, method='equi', memory=None)

    else: # knockoff diagnostic
        adjust_marg=not configs['disable_adjust_marg']
        if method_diagn_gen == 'dagma':
            _configs = configs.copy()
            _configs['gen_W'] = 'torch'
            W_est_no_filter, _, _ = utils.fit(X, _configs, original=True)
            preds = X @ W_est_no_filter
        else:
            p = configs['d']
            preds = np.array(Parallel(n_jobs=n_jobs)(delayed(
            _get_single_clf_ko)(X, j, method_diagn_gen, lasso_alpha, PLS_n_comp, device) for j in tqdm(range(p))))
            preds = preds.T
        
        X_tildes = conditional_sequential_gen_ko(X, preds, n_jobs=n_jobs, discrete=False, adjust_marg=adjust_marg)

    return X_tildes

#########################
# Utilities 
#########################

def _get_fitting_model(X_input, X_target, method, alpha, n_comp, device):
    p = X_input.shape[1] + 1

    if method == "lasso":
        if alpha == 'knockoff_diagn':
            lambda_max = np.max(np.abs(np.dot(X_input.T, X_target))) / (2 * (p - 1))
            alpha = (lambda_max / 100)
            clf = Lasso(alpha)
        elif alpha == 'sklearn':
            clf = Lasso()
        elif alpha == 'OLS':
            clf = Lasso(alpha=0)

    if method == "OLS_cuda":
        clf = LinearRegression()
    
    if method == "logreg_cv":
        clf = LogisticRegressionCV(cv=5, max_iter=int(10e4), n_jobs=-1)

    if method == "xgb":
        # clf = xgb.XGBRegressor(n_jobs=-1)
        clf = xgb.XGBRegressor(device=device)

    if method == 'elastic':
        clf = ElasticNet()

    if method == 'PLS':
        clf = PLSRegression(n_components=n_comp)

    return clf

def _get_single_clf_ko(X, j, method="lasso", alpha='knockoff_diagn', n_comp=2, device='cpu'):
    """
    Fit a single classifier to predict the j-th variable from all others.

    Args:
        X : input data
        j (int): variable index
        method (str, optional): Classifier used. Defaults to "lasso".
        all_X: if True, all X is used to fit j rather than only X[i] where i != j.

    Returns:
        pred: Predicted values for variable j from all others.
    """
    
    n, p = X.shape
    idc = np.array([i for i in np.arange(0, p) if i != j])

    clf = _get_fitting_model(X[:, idc], X[:, j], method, alpha, n_comp, device)
    clf.fit(X[:, idc], X[:, j])
    pred = clf.predict(X[:, idc])
    return pred

def _get_single_clf(X_input, X_target, method="lasso", alpha='knockoff_diagn', n_comp=2, device='cpu'):
    clf = _get_fitting_model(X_input, X_target, method, alpha, n_comp, device)

    if "cuda" in method:
        X_input = cp.asarray(X_input)
        X_target = cp.asarray(X_target)
    clf.fit(X_input, X_target)
    pred = clf.predict(X_input)
    if "cuda" in method:
        pred = pred.get()
    return pred

def _estimate_distribution(X, shrink=True, cov_estimator='ledoit_wolf', n_jobs=1):
    """
    Adapted from hidimstat: https://github.com/ja-che/hidimstat
    """
    alphas = [1e-3, 1e-2, 1e-1, 1]

    mu = X.mean(axis=0)
    Sigma = empirical_covariance(X)

    if shrink or not _is_posdef(Sigma):

        if cov_estimator == 'ledoit_wolf':
            Sigma_shrink = ledoit_wolf(X, assume_centered=True)[0]

        elif cov_estimator == 'graph_lasso':
            model = GraphicalLassoCV(alphas=alphas, n_jobs=n_jobs)
            Sigma_shrink = model.fit(X).covariance_

        else:
            raise ValueError('{} is not a valid covariance estimated method'
                             .format(cov_estimator))

        return mu, Sigma_shrink

    return mu, Sigma


def _is_posdef(X, tol=1e-14):
    """Check a matrix is positive definite by calculating eigenvalue of the
    matrix. Adapted from hidimstat: https://github.com/ja-che/hidimstat

    Parameters
    ----------
    X : 2D ndarray, shape (n_samples x n_features)
        Matrix to check

    tol : float, optional
        minimum threshold for eigenvalue

    Returns
    -------
    True or False
    """
    eig_value = np.linalg.eigvalsh(X)
    return np.all(eig_value > tol)



def _adjust_marginal(v, ref, discrete=False): # TODO(jiahang): what's this?
    """
    Make v follow the marginal of ref.
    """
    if discrete:
        sorter = np.argsort(v)
        sorter_ = np.argsort(sorter)
        return np.sort(ref)[sorter_]
    
    else:
        G = ECDF(ref)
        F = ECDF(v)

        unif_ = F(v)

        # G_inv = np.argsort(G(ref))
        G_inv = monotone_fn_inverter(G, ref)

        return G_inv(unif_)


def _get_samples_ko(X, pred, j, discrete=False, adjust_marg=True):
    """

    Generate a Knockoff for variable j.

    Args:
        X : input data
        pred (array): Predicted Xj using all other variables
        j (int): variable index
        discrete (bool, optional): Indicates discrete or continuous data. Defaults to False.
        adjust_marg (bool, optional): Whether to adjust marginals or not. Defaults to True.

    Returns:
        sample: Knockoff for variable j.
    """
    n, p = X.shape

    residuals = X[:, j] - pred
    indices_ = np.arange(residuals.shape[0])
    np.random.shuffle(indices_)

    """
    TODO(jiahang):
    In codes, this shuffle is applied to the sample dimension, not the feature dimension
    different from the paper.
    but very similar to permutation knockoff initially.
    """
    sample = pred + residuals[indices_]

    if adjust_marg:
        sample = _adjust_marginal(sample, X[:, j], discrete=discrete)

    return sample[np.newaxis].T


def conditional_sequential_gen_ko(X, preds, n_jobs=1, discrete=False, adjust_marg=True):
    """
    Generate Knockoffs for all variables in X.

    Args:
        X : input data
        preds (array): Predicted values for all variables
        n_jobs (int, optional): Number of parallel jobs. Defaults to 1.
        discrete (bool, optional): Indicates discrete or continuous data. Defaults to False.
        adjust_marg (bool, optional): Whether to adjust marginals or not. Defaults to True.
    Returns:
        samples: Knockoffs for all variables in X.
    """
    n, p = X.shape

    samples = np.hstack(Parallel(n_jobs=n_jobs)(delayed(
        _get_samples_ko)(X, preds[:, j], j, discrete=discrete, adjust_marg=adjust_marg) for j in tqdm(range(p))))
    
    return samples



