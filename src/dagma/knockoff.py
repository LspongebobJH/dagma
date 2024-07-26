import numpy as np
from knockoff_gan import KnockoffGAN
from deep_knockoff.machine import KnockoffMachine
from deep_knockoff.parameters import GetTrainingHyperParams, SetFullHyperParams
from deep_knockoff.gaussian import GaussianKnockoffs
from knockoff_diagn import get_knockoffs_stats
import utils

def knockoff(X : np.ndarray, configs):
    n = configs['n']
    d = configs['d']
    knock_type = configs['knock_type']
    if 'seed' in configs.keys():
        seed = configs['seed']
    else:
        seed = configs['seed_knockoff']

    if knock_type == 'permutation':
        X_tilde = np.zeros_like(X)
        for seed, col in enumerate(range(X.shape[1])):
            rng = np.random.default_rng(seed=seed)
            X_tilde[:, col] = rng.permutation(X[:, col])

    elif knock_type == 'knockoff_gan':
        # TODO: tensorflow fail running on GPU, but not slow now.
        niter = configs['niter']
        norm_tag = configs['norm_knockoffGAN']
        X = utils.norm(X) if norm_tag else X
        X_tilde = KnockoffGAN(x_train = X, x_name = 'Normal', niter = niter)

    elif knock_type == 'knockoff_diagn':
        X_tilde = get_knockoffs_stats(X, configs)

    elif knock_type == 'deep_knockoff':
        SigmaHat = np.cov(X, rowvar=False)
        second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(X,0), method="sdp")
        corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)
        training_params = GetTrainingHyperParams('gaussian')
        pars = SetFullHyperParams(training_params, n, d, corr_g)
        
        checkpoint_name = "checkpoints/deep_knockoff/" + 'gaussian'
        logs_name = "logs/log_3_model"

        machine = KnockoffMachine(pars, checkpoint_name=checkpoint_name, logs_name=logs_name)

        print("fitting deep knockoff")
        machine.train(X)
        X_tilde = machine.generate(X)

    elif knock_type == 'standard': # not work yet
        mean = np.mean(X, axis=0)
        cov = np.cov(X.T)
        inv_cov = np.linalg.inv(cov)
        s = np.ones(d) # TODO: should be customized
        diag_s = np.diag(s)
        new_mean = (X - diag_s @ inv_cov @ X.T).T
        new_cov = 2 * diag_s - diag_s @ inv_cov @ diag_s
        X_tilde = np.random.multivariate_normal(new_mean, new_cov, size=n)

    return X_tilde
