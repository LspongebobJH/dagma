#!/bin/bash

n_list=(60 80 100)
options=(9)
for n in "${n_list[@]}"; do
    for option in "${options[@]}"; do
        s0=$(( n * 6 ))
        mkdir ~/dagma/src/dagma/simulated_data/v47_repr/v${n}_${s0}_normX_sym1_option_${option}_OLS_X
        ln -snf ~/dagma/src/dagma/simulated_data/v47/v${n}_${s0}_normX_sym1_option_9_OLS/X \
            ~/dagma/src/dagma/simulated_data/v47_repr/v${n}_${s0}_normX_sym1_option_9_OLS_X/X
        # ln -snf ~/dagma/src/dagma/simulated_data/v47/v${n}_${s0}_normX_sym1_option_9_OLS/knockoff \
        #     ~/dagma/src/dagma/simulated_data/v47_repr/v${n}_${s0}_normX_sym1_option_9_OLS/knockoff
            
    done
done