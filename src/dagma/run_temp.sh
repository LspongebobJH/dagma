#!/bin/bash

alpha_list=(10 1 0.1 0.05 0.01 0)
l1_ratio_list=(0.1 0.5 0.9)

for alpha in "${alpha_list[@]}"; do
    for l1_ratio in "${l1_ratio_list[@]}"; do
        mv /home/jiahang/dagma/src/dagma/simulated_data/v49/v100_600_normX=sym1_option=5_knock=PLS_model=L1+L2_alpha=${alpha}_l1_ratio=${l1_ratio} \
            /home/jiahang/dagma/src/dagma/simulated_data/v49/v100_600_normX=sym1_option=5_knock=OLS_cuda_model=L1+L2_alpha=${alpha}_l1_ratio=${l1_ratio}
    done
done