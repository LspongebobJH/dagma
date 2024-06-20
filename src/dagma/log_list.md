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

* log_16: 2000 samples, 20 nodes, 80 edges, type_3_global (when using dag control, then become type_4_global),
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

* log_1000: 2000 samples, 20 nodes, 120 edges, type_4, W_torch
** dag_1: fdr 0.0778±0.0280 | power 0.6083±0.0811
** dag_3: fdr 0.2469±0.0148 | power 0.8175±0.0058
** dag_5: fdr 0.7820±0.2429 | power 0.0675±0.0807
** dag_7: fdr 0.2146±0.0615 | power 0.8158±0.0079

* log_1001: 2000 samples, 20 nodes, 120 edges, type_4_global, W_torch
** dag_1: 