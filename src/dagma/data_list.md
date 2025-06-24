v1: deprecated
v2: 2000 samples, 20 nodes, 80 edges, 10 knockoff seeds
v3: 2000 samples, 20 nodes, 50 edges, 10 knockoff seeds
v4: deprecated
v5: 2000 samples, 20 nodes, 120 edges, 10 knockoff seeds
v6: 2000 samples, 20 nodes, 180 edges, 10 knockoff seeds
v7: 2000 samples, 60 nodes, 1000 edges, 10 knockoff seeds
v8: 2000 samples, 20 nodes, 120 edges, 10 knockoff seeds, W_torch, compared with v5 (W).
v10: 2000 samples, 20 nodes, 80 edges, 10 knockoff seeds, W_torch, compared with v2 (W).

v12: (deprecated) sweep, 2000 samples, [40, 60] nodes, [n * 4] edges, 10 knockoff seeds by with larger training epochs (2000 -> 5000), W_torch, version index is the n_nodes. Note that version uses X of v11 (soft links).
v13: sweep, 2000 samples, [40, 60, 80, 100] nodes, [n * 4] edges, 10 knockoff seeds with max_abs_col norm, W_torch, version index is the n_nodes. Note that version uses X of v11 (soft links).
v14: sweep, 2000 samples, [40, 60, 80, 100] nodes, [n * 4] edges, 10 knockoff seeds with max_abs_col norm, max_abs_col norm for X_all before DAGMA fitting, W_torch, version index is the n_nodes. Note that version uses X of v11 (soft links), and knockoff of v13 (softlinks)
v15: sweep, [40, 60] nodes, ..., deconv within dagma, deconv_1, order 5, X and knockoff both from v11
    v16: sweep, [40, 60] nodes, ..., deconv within dagma, deconv_1, order 3, X and knockoff both from v11
    v17: sweep, [40, 60] nodes, ..., deconv within dagma, deconv_2, no order def, X and knockoff both from v11
v18, v19, v20, v21, v22: deprecated, see logs
v23: deconv in dagma, deconv_4, actually no deconv. This is not for any experimental purpose. It's just for testing whether the new formulation of deconv_4 can achieve the same performance as baseline of v11.
v24: deconv in dagma, deconv_4_1, alpha 0.3, order 2
v25: deconv in dagma, deconv_4_2, alpha 0.3
v26: deconv in dagma, deconv_4_1, alpha 0.3, order 2, warm_iter 5e4 -> 8e4
v27: deconv in dagma, deconv_4_2, alpha 0.3, warm_iter 5e4 -> 8e4
v28: deconv in dagma, deconv_4_1, alpha=0.3, order=2, add loss term ||X - XW_dir||
v29: deconv in dagma, deconv_4_2, alpha=0.3, order=2, add loss term ||X - XW_dir||
v30: deconv in dagma, deconv_4_1, alpha=0.1, order=2, add loss term ||X - XW_dir||
v31: deconv in dagma, deconv_4_2, alpha=0.1, order=2, add loss term ||X - XW_dir||
v32: based on v11, increase the number of T to make DAG loss converge better and see if it can resolve our problems.
v36: (ER4, knockoffDiagnosis)
    2000 samples, [20, 40, 60, 80, 100, 120, 140, 160, 180, 200] nodes, [n * 4] edges, 10 knockoff seeds, W_torch, ..., disable adjust_marg

v33: (ER5/6, knockoffGAN, only X is useful)
    2000 samples, [10, 20, 40, 60, 80, 100] nodes, [n * 5 / 6] edges, 5 knockoff seeds, W_torch, version index is the n_nodes.
v34: (ER4, knockoffDiagnosis)
    2000 samples, [20, 40, 60, 80, 100, 120, 140, 160, 180, 200] nodes, [n * 4] edges, 10 knockoff seeds, W_torch, X from v11
v38: (ER5/6, knockoffDiagnosis)
    2000 samples, [60, 80] nodes, [n * 5/6] edges, 5 knockoff seeds, W_torch, ..., X from v33

v35: (ER5/6, knockoffDiagnosis)
    2000 samples, [60, 80] nodes, [n * 5 / 6] edges, 5 knockoff seeds, W_torch, ...
v37: test whether disable "remove diagonal of W21 and W12 blocks" will impact the Z distribution shift



v40: (ER4, transitive effect experiments, test.py when exp_group_idx == v40)
v41: (ER4, in-degree experiments, exp 1, gen_copies.py when --method_diagn_gen=lasso)
    one problem: "ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.269e+01, tolerance: 2.063e-01." Meaning that some features cannot be regressed well.

    v41 others: name with alpha_skleanr: --method_diagn_gen=lasso + --lasso_alpha=sklearn
    v41 others: name with xgb: --method_diagn_gen=xgb

v42: (ER4, in-degree experiments, exp 2, test.py when exp_group_idx == v42)
v43: (ER4, relation experiments, test.py when exp_group_idx == v43)
v44_deprecated: (ER4, markov blanket experiments, test.py when exp_group_idx == v44)

=== common used datasets below ===
v11: (ER4, knockoffGAN, only X is useful) 
    2000 samples, [10, 20, 40, 60, 80, 100] nodes, [n * 4] edges, 10 knockoff seeds, W_torch, version index is the n_nodes.
- condX_n: only seeds of X with condition number of X < n
- normX_topoCol: when --norm_data_gen = topo_col in gen_copies.py, basically each x is normalized after being generated.
- normX_B1Col: when --norm_data_gen = B_1_Col in gen_copies.py, basically let W_true /= B_true.sum(axis=0), col norm. only
on seeds with cond > 5e4

v39: (ER4, original DAGMA, dagma_torch.py)
    2000 samples, [80, 100], [n * 4] edges, 2 knockoff seeds, W_torch, ..., X from v11

v44 (and v45 ): try different options of knockoff, such as LASSO / OLS / XGB for knockoff fitting, and all nodes / 
parents / ancestors as input variables for knockoff fitting

v46: only focus on option_5_OLS.
v47: based on W_true in [-1, 1] (workable simulations), try different options of knockoff fitting
v48: original genie3 and grnboost2 fitting W of simulated data
v49: based on W_true in [-1, 1] (workable simulations), fit [X, X_tilde] with genie3 and grnboost2

v50: original NOTEARS on W_true in [-1, 1] of simulated data
v51: with knockoff fitting NOTEARS on W_true in [-1, 1] of simulated data

v52: original golem on W_true in [-1, 1] of simulated data
v53: with knockoff fitting golem on W_true in [-1, 1] of simulated data
- note that some PLS results have problems

v54: original dag_gnn
v55: dag_gnn + knockoff

=== common used options ===

topo_sort: only available to 10, affecting generation order of existing knockoff according to topological sort
    of DAG version of the latent graph.
Wdagma: use W obtained from DAGMA rather than genie3 (grnboost2) for topological sort
new: original knockoff generation is, considering X (n*p) as a whole design matrix, generating X_tilde for X.
    Since genie3 models causal graph modeling as p separate regression task, then each task has its own design
    matrix. The new knockoff generation is to generate X_tilde (n*(p-1)) for each task, thus have p*n*(p-1) X_tilde.
nComp: n_component of PLS
OLS, PLS: which model to fit j_tilde
disable_norm: disable normalization of genie3 and grnboost2 for target values
disable_remove_self: after the old style knockoff generation, model fitting will remove X_self and X'_self to fit X_self. Now we don't remove self. Instead, we use ALL to fit X_self and remove them in resulted W.
options:
- 5: all other nodes / j fit j_tilde
- 10: all other nodes / j + existing knockoff fit j_tilde
