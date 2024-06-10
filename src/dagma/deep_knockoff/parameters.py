def GetDistributionParams(model,p):
    """
    Returns parameters for generating different data distributions
    """
    params = dict()
    params["model"] = model
    params["p"] = p
    if model == "gaussian":
        params["rho"] = 0.5
    elif model == "gmm":
        params["rho-list"] = [0.3,0.5,0.7]
    elif model == "mstudent":
        params["df"] = 3
        params["rho"] = 0.5
    elif model == "sparse":
        params["sparsity"] = int(0.3*p)
    else:
        raise Exception('Unknown model generating distribution: ' + model)
    
    return params
        
def GetTrainingHyperParams(model):
    """
    Returns the default hyperparameters for training deep knockoffs
    as described in the paper
    """
    params = dict()
    
    params['GAMMA'] = 1.0
    if model == "gaussian":
        params['LAMBDA'] = 1.0
        params['DELTA'] = 1.0
    elif model == "gmm":
        params['LAMBDA'] = 1.0
        params['DELTA'] = 1.0
    elif model == "mstudent":
        params['LAMBDA'] = 0.01
        params['DELTA'] = 0.01
    elif model == "sparse":
        params['LAMBDA'] = 0.1
        params['DELTA'] = 1.0
    else:
        raise Exception('Unknown data distribution: ' + model)
        
    return params

def GetFDRTestParams(model):
    """
    Returns the default hyperparameters for performing controlled
    variable selection experiments as described in the paper
    """
    params = dict()
    # Test parameters for each model
    if model in ["gaussian", "gmm"]:
        params["n"] = 150
        params["elasticnet_alpha"] = 0.1
    elif model in ["mstudent"]:
        params["n"] = 200
        params["elasticnet_alpha"] = 0.0
    elif model in ["sparse"]:
        params["n"] = 200
        params["elasticnet_alpha"] = 0.0
    
    return params

def SetFullHyperParams(training_params, n, p, corr_g):
    """
    use hyperparameters of experiment-1.ipynb in original github
    """
    # Set the parameters for training deep knockoffs
    pars = dict()
    # Number of epochs
    # pars['epochs'] = 10
    pars['epochs'] = 100
    # Number of iterations over the full data per epoch
    # pars['epoch_length'] = 50
    pars['epoch_length'] = 100
    # Data type, either "continuous" or "binary"
    pars['family'] = "continuous"
    # Dimensions of the data
    pars['p'] = p
    # Size of the test set
    # pars['test_size']  = int(0.1*n)
    pars['test_size']  = 0
    # Batch size
    # pars['batch_size'] = int(0.45*n)
    pars['batch_size'] = int(0.5*n)
    # Learning rate
    pars['lr'] = 0.01
    # When to decrease learning rate (unused when equal to number of epochs)
    pars['lr_milestones'] = [pars['epochs']]
    # Width of the network (number of layers is fixed to 6)
    pars['dim_h'] = int(10*p)
    # Penalty for the MMD distance
    pars['GAMMA'] = training_params['GAMMA']
    # Penalty encouraging second-order knockoffs
    pars['LAMBDA'] = training_params['LAMBDA']
    # Decorrelation penalty hyperparameter
    pars['DELTA'] = training_params['DELTA']
    # Target pairwise correlations between variables and knockoffs
    pars['target_corr'] = corr_g
    # Kernel widths for the MMD measure (uniform weights)
    pars['alphas'] = [1.,2.,4.,8.,16.,32.,64.,128.]

    return pars