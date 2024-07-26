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

=== common used datasets below ===
v11: (baseline, ER4, knockoffGAN) 
    2000 samples, [10, 20, 40, 60, 80, 100] nodes, [n * 4] edges, 10 knockoff seeds, W_torch, version index is the n_nodes.
v33: (denser, ER5/6, knockoffGAN)
    2000 samples, [10, 20, 40, 60, 80, 100] nodes, [n * 5 / 6] edges, 5 knockoff seeds, W_torch, version index is the n_nodes.
v34: (ER4, knockoffDiagnosis)
    2000 samples, [20, 40, 60, 80, 100, 120, 140, 160, 180, 200] nodes, [n * 4] edges, 10 knockoff seeds, W_torch, ...
v35: (ER5/6, knockoffDiagnosis)
    2000 samples, [60, 80] nodes, [n * 5 / 6] edges, 5 knockoff seeds, W_torch, ...
    

