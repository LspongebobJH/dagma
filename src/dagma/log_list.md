* log_1 (deprecated): 2000 samples, 20 nodes, 50 edges, type_3_global, expected fdr 0.2, real fdr 0.0928±0.0362, ALL DAG
* log_2 (deprecated): ..., type_3 (column-wise), ..., real fdr 0.1509±0.0948, 1 NOT DAG
* log_3: 2000 samples, 20 nodes, 50 edges, type_3_global, expected fdr 0.2, real fdr 0.0928±0.0362, power 0.9640±0.0215, ALL DAG
* log_4: 2000 samples, 20 nodes, 50 edges, type_3, expected fdr 0.2, real fdr 0.1509±0.0948, power 0.5120±0.0392, 1 NOT DAG
* log_5: based on log_3, but 80 edges, fdr 0.2669±0.1724 | power 0.9663±0.0112, 2 NOT DAG
* log_6: based on log_4, but 80 edges, fdr 0.4864±0.0529 | power 0.7725±0.0406, ALL NOT DAG
* log_7 (deprecated): same as log_3
* log_8 (deprecated): same as log_4
* log_11: based on log_3, but 120 edges, fdr 0.3168±0.0911 | power 0.9475±0.0075, 3 NOT DAG
* log_12: based on log_4, but 120 edges, fdr 0.4260±0.0491 | power 0.8375±0.0172, ALL NOT DAG
* log_13 (deprecated): based on log_3, but 80 edges, dag test first (dag_1), fdr 0.2092±0.1017 | power 0.9625±0.0000
* log_14 (deprecated): based on log_4, but 120 edges, dag test first (dag_2), fdr 0.2750±0.0095 | power 0.9442±0.0053

* log_15: 2000 samples, 20 nodes, 120 edges, type_3_global (when using dag control, then become type_4_global),
expected fdr 0.2, ALL DAG
** _1: dag_1, first remove Z from min to max, until dag, then fdr control.
*** fdr 0.2796±0.0058 | power 0.9467±0.0041
** _2: dag_2, first include Z from max to min, until not dag, then fdr control.
*** same as _1
** _3: dag_3, first fdr control, then remove Z from min to max, until dag.
*** fdr 0.2750±0.0095 | power 0.9442±0.0053
** _4: dag_4, first fdr control, then include Z from max to min, until not dag.
*** same as _3
** _5(deprecated -> _7): dag_5, first fdr control, then remove Q from min to max, until dag.
*** fdr 0.4038±0.2049 | power 0.6783±0.4060
** _6(deprecated -> _8): dag_6, first fdr control, then include Q from max to min, until not dag.
*** same as _5
** _9: dag_7, first remove W from min to max, until dag, then fdr control

* log_16 (deprecated): 2000 samples, 20 nodes, 80 edges, type_3_global (when using dag control, then become type_4_global),
expected fdr 0.2, ALL DAG
** _1: dag_1, first remove Z from min to max, until dag, then fdr control.
*** fdr 0.3880±0.0149 | power 0.9625±0.0000
** _2: dag_2, first include Z from max to min, until not dag, then fdr control.
*** same as _1
** _3: dag_3, first fdr control, then remove Z from min to max, until dag.
*** fdr 0.2341±0.1083 | power 0.9625±0.0000
** _4: dag_4, first fdr control, then include Z from max to min, until not dag.
*** same as _3
** _5(deprecated -> _7): dag_5, first fdr control, then remove Q from min to max, until dag.
*** fdr 0.2822±0.2037 | power 0.7800±0.3650
** _6(deprecated -> _8): dag_6, first fdr control, then include Q from max to min, until not dag.
*** same as _5
** _9: dag_7, first remove W from min to max, until dag, then fdr control

* log_17: 2000 samples, 20 nodes, 80 edges, type_4_global
* log_18: 2000 samples, 20 nodes, 120 edges, type_4_global
* log_19: 2000 samples, 20 nodes, 80 edges, type_4
* log_20: 2000 samples, 20 nodes, 120 edges, type_4
* log_21: 2000 samples, 20 nodes, 180 edges, type_4
* log_22: 2000 samples, 20 nodes, 180 edges, type_4_global
* log_23: 2000 samples, 20 nodes, 180 edges, type_3
* log_24: 2000 samples, 20 nodes, 180 edges, type_3_global
* log_25: 2000 samples, 60 nodes, 1000 edges, type_4
* log_26: 2000 samples, 60 nodes, 1000 edges, type_4_global
* log_27: 2000 samples, 60 nodes, 1000 edges, type_3
* log_28: 2000 samples, 60 nodes, 1000 edges, type_3_global

* log_29 (data version v11): 
sweep, 2000 samples, [10, 40, 60, 80, 100] nodes, [n * 4] edges, W_torch, type_3
name is the n_nodes.

* log_30 (data version v11): 
sweep, 2000 samples, [10, 40, 60, 80, 100] nodes, [n * 4] edges, W_torch, type_3_global
name is the n_nodes.

* log_31 (data version v11): 
sweep, 2000 samples, [10, 40, 60, 80, 100] nodes, [n * 4] edges, W_torch, type_4, dag_1
name is the n_nodes.

* log_32 (data version v11): 
sweep, 2000 samples, [10, 40, 60, 80, 100] nodes, [n * 4] edges, W_torch, type_4_global, dag_1
name is the n_nodes.

* log_33 (data version v12)(vs log_29):
sweep, 2000 samples, [40, 60, 80] nodes, [n * 4] edges, W_torch, type_3, name is the n_nodes.
knockoffGAN niter 2000 -> 5000

* log_34 (data version v12)(vs log_30):
sweep, 2000 samples, [40, 60, 80] nodes, [n * 4] edges, W_torch, type_3_global, name is the n_nodes.
knockoffGAN niter 2000 -> 5000

* log_35 (data version v13)(vs log_29):
sweep, 2000 samples, [40, 60, 80] nodes, [n * 4] edges, W_torch, type_3, name is the n_nodes.
knockoffGAN with max_col_abs normalization
check the param:norm in knockoff configs since this param is not set up in other configs.

* log_36 (data version v13)(vs log_30):
sweep, 2000 samples, [40, 60, 80] nodes, [n * 4] edges, W_torch, type_3_global, name is the n_nodes.
knockoffGAN with max_col_abs normalization
check the param:norm in knockoff configs since this param is not set up in other configs.

* log_37 (data version v11)(vs log_29):
sweep, 2000 samples, [40] nodes, [n * 4] edges, W_torch, type_3, name is the n_nodes.
network deconv (deconv_1)

* log_38 (data version v11)(vs log_30):
sweep, 2000 samples, [40] nodes, [n * 4] edges, W_torch, type_3_global, name is the n_nodes.
network deconv (deconv_1)

* log_39 (data version v11)(vs log_29):
sweep, 2000 samples, [40, 60, 80] nodes, [n * 4] edges, W_torch, type_3, name is the n_nodes.
network deconvolution (deconv_2)

* log_40 (data version v11)(vs log_30):
sweep, 2000 samples, [40, 60, 80] nodes, [n * 4] edges, W_torch, type_3_global, name is the n_nodes.
network deconvolution (deconv_2)

* log_41 (data version v11)(vs log_29):
sweep, 2000 samples, [40, 60, 80] nodes, [n * 4] edges, W_torch, type_3, name is the n_nodes.
network deconvolution (deconv_2) + diag removal

* log_42 (data version v11)(vs log_30):
sweep, 2000 samples, [40, 60, 80] nodes, [n * 4] edges, W_torch, type_3_global, name is the n_nodes.
network deconvolution (deconv_2) + diag removal

* log_43 (data version v14)(vs log_35):
sweep, 2000 samples, [40, 60] nodes, [n * 4] edges, W_torch, type_3, name is the n_nodes.
knockoffGAN with max_col_abs normalization + max_col_bas norm for X_all before DAGMA fitting
same X and knockoff from v13 (soft link), but different W

* log_44 (data version v14)(vs log_36):
sweep, 2000 samples, [40, 60] nodes, [n * 4] edges, W_torch, type_3_global, name is the n_nodes.
knockoffGAN with max_col_abs normalization + max_col_bas norm for X_all before DAGMA fitting.
same X and knockoff from v13 (soft link), but different W

log_45 (data version v11)(vs log_39)
sweep, 2000 samples, [40, 60, 80] nodes, [n * 4] edges, W_torch, type_3, name is the n_nodes.
network deconvolution (deconv_2), DAG control on W before deconv

log_46 (data version v11)(vs log_40)
sweep, 2000 samples, [40, 60, 80] nodes, [n * 4] edges, W_torch, type_3_global, name is the n_nodes.
network deconvolution (deconv_2), DAG control on W before deconv




* log_1000 (local): 2000 samples, 20 nodes, 120 edges, type_4, W_torch
* log_1001 (local): 2000 samples, 20 nodes, 120 edges, type_4_global, W_torch
* log_1002 (local): 2000 samples, 20 nodes, 80 edges, type_4, W_torch
* log_1003 (local): 2000 samples, 20 nodes, 80 edges, type_4_global, W_torch
* log_1004 (local): 2000 samples, 20 nodes, 80 edges, type_4, W_torch, no dag control (same as type_3)
* log_1005 (local): 2000 samples, 20 nodes, 80 edges, type_4_global, W_torch, no dag control (same as type_3_global)
* log_1006 (local): 2000 samples, 20 nodes, 120 edges, type_4, W_torch, no dag control (same as type_3)
* log_1007 (local): 2000 samples, 20 nodes, 120 edges, type_4_global, W_torch, no dag control (same as type_3_global)

