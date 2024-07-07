v1: deprecated
v2: 2000 samples, 20 nodes, 80 edges, 10 knockoff seeds
v3: 2000 samples, 20 nodes, 50 edges, 10 knockoff seeds
v4: deprecated
v5: 2000 samples, 20 nodes, 120 edges, 10 knockoff seeds
v6: 2000 samples, 20 nodes, 180 edges, 10 knockoff seeds
v7: 2000 samples, 60 nodes, 1000 edges, 10 knockoff seeds
v8: 2000 samples, 20 nodes, 120 edges, 10 knockoff seeds, W_torch, compared with v5 (W).
v10: 2000 samples, 20 nodes, 80 edges, 10 knockoff seeds, W_torch, compared with v2 (W).

=== common used datasets below ===
v11: sweep, 2000 samples, [10, 40, 60, 80, 100] nodes, [n * 4] edges, 10 knockoff seeds, W_torch, version index is the n_nodes.
v12: (deprecated) sweep, 2000 samples, [40, 60] nodes, [n * 4] edges, 10 knockoff seeds by with larger training epochs (2000 -> 5000), W_torch, version index is the n_nodes. Note that version uses X of v11 (soft links).
v13: sweep, 2000 samples, [40, 60, 80, 100] nodes, [n * 4] edges, 10 knockoff seeds with max_abs_col norm, W_torch, version index is the n_nodes. Note that version uses X of v11 (soft links).
v14: sweep, 2000 samples, [40, 60, 80, 100] nodes, [n * 4] edges, 10 knockoff seeds with max_abs_col norm, max_abs_col norm for X_all before DAGMA fitting, W_torch, version index is the n_nodes. Note that version uses X of v11 (soft links), and knockoff of v13 (softlinks)
v15: sweep, [40, 60] nodes, ..., deconv within dagma, deconv_1, order 5, X and knockoff both from v11
    v16: sweep, [40, 60] nodes, ..., deconv within dagma, deconv_1, order 3, X and knockoff both from v11
