* log_1: 2000 samples, 20 nodes, 50 edges, type_3_global, expected fdr 0.2, real fdr 0.0928±0.0362, ALL DAG
* log_2: ..., type_3 (column-wise), ..., real fdr 0.1509±0.0948, 1 NOT DAG
* log_3: same as log_1, add calculate power 0.9640±0.0215
* log_4: same as log_2, add calculate power 0.5120±0.0392
* log_5: based on log_3, but 80 edges, fdr 0.2669±0.1724 | power 0.9663±0.0112, ALL DAG
* log_6: based on log_4, but 80 edges, fdr 0.4864±0.0529 | power 0.7725±0.0406, ALL NOT DAG
* log_7 (deprecated): same as log_3
* log_8 (deprecated): same as log_4
* log_11: based on log_3, but 120 edges, fdr 0.3168±0.0911 | power 0.9475±0.0075, 3 NOT DAG
* log_12: based on log_3, but 120 edges, fdr 0.4260±0.0491 | power 0.8375±0.0172, ALL NOT DAG