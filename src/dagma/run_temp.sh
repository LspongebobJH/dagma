#!/bin/bash

n_list=(20 40 60 80 100)
options=(9)
for n in "${n_list[@]}"; do
    for option in "${options[@]}"; do
        s0=$(( n * 6 ))
        mv /home/jiahang/dagma/src/dagma/simulated_data/v47/v${n}_${s0}_normX_sym1_option_13_PLS_dedup /home/jiahang/dagma/src/dagma/simulated_data/v47/deprecated
        mv /home/jiahang/dagma/src/dagma/simulated_data/v47/v${n}_${s0}_normX_sym1_option_13_PLS_dedup_topo_sort /home/jiahang/dagma/src/dagma/simulated_data/v47/deprecated

        mv /home/jiahang/dagma/src/dagma/logs/log_99/log_47_${n}_${s0}_normX_sym1_option_13_PLS_dedup_1-10.brief /home/jiahang/dagma/src/dagma/logs/log_99/deprecated
        mv /home/jiahang/dagma/src/dagma/logs/log_99/log_47_${n}_${s0}_normX_sym1_option_13_PLS_dedup_1-10 /home/jiahang/dagma/src/dagma/logs/log_99/deprecated
        mv /home/jiahang/dagma/src/dagma/logs/log_99/log_47_${n}_${s0}_normX_sym1_option_13_PLS_dedup_topo_sort_1-10.brief /home/jiahang/dagma/src/dagma/logs/log_99/deprecated
        mv /home/jiahang/dagma/src/dagma/logs/log_99/log_47_${n}_${s0}_normX_sym1_option_13_PLS_dedup_topo_sort_1-10 /home/jiahang/dagma/src/dagma/logs/log_99/deprecated
            
    done
done