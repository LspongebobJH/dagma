#!/bin/bash

# version=11
# s0=80
# d=20
# n=2000

########################################
# Generating X
########################################

# python gen_copies.py --gen_type X --n $n --s0 $s0 --d $d \
# --version $d --root_path simulated_data/v${version} &

########################################
# Generating X, sweep over n_nodes
########################################

# n=2000
# nodes=(20 40 60 80)
# for d in "${nodes[@]}"; do
#     s0=$(( d * 5 ))
#     python gen_copies.py --gen_type X \
#     --n $n --d $d --s0 $s0 \
#     --root_path simulated_data/v33 \
#     --version ${d}_${s0} &
# done

# wait

# nodes=(20 40 60 80)
# for d in "${nodes[@]}"; do
#     s0=$(( d * 6 ))
#     python gen_copies.py --gen_type X \
#     --n $n --d $d --s0 $s0 \
#     --root_path simulated_data/v33 \
#     --version ${d}_${s0} &
# done
    

########################################
# Generating Knockoff
########################################

# for i in {1..10}; do
#     python gen_copies.py --gen_type knockoff --n $n --s0 $s0 --d $d \
#     --seed_knockoff $i --version $d --root_path simulated_data/v${version} &
# done

# wait

#########################################
# Generating Knockoff, sweep over n_nodes
#########################################

# data_option=X
# dst_data_version=41
# # dst_final_version=20
# src_data_version=11
# # src_final_version=20
# n=2000
# cuda_idx=5
# nodes=(40 80 100)
# for d in "${nodes[@]}"; do
    
#     s0=$(( d * 4 ))
#     version=${d}
#     ./create_data_dir.sh $data_option $dst_data_version $version $src_data_version $version

#     # for i in {1..5}; do
#     for i in {1..3}; do
#         python gen_copies.py --gen_type knockoff --knock_type knockoff_diagn \
#         --n $n --d $d --s0 $s0 --seed_knockoff $i --method_diagn_gen=lasso \
#         --root_path simulated_data/v${dst_data_version} \
#         --version $version --device cuda:${cuda_idx} \
#         >/home/jiahang/dagma/src/dagma/simulated_data/v${dst_data_version}/v$version/knockoff/log_${i} 2>&1 &
#     done

#     cuda_idx=$(( cuda_idx + 1 ))

#     # for i in {6..10}; do
#     #     python gen_copies.py --gen_type knockoff --knock_type knockoff_diagn \
#     #     --n $n --d $d --s0 $s0 --seed_knockoff $i \
#     #     --root_path simulated_data/v${dst_data_version} \
#     #     --version $version --device cuda:7 &
#     # done

#     # wait

# done

#########################################
# Misc
#########################################

# python dagma_torch.py --d 10 --seed 1 --device cuda:0 &
# python dagma_torch.py --d 10 --seed 2 --device cuda:1 &
# python dagma_torch.py --d 20 --seed 1 --device cuda:2 &
# python dagma_torch.py --d 20 --seed 2 --device cuda:3 &
# python dagma_torch.py --d 40 --seed 1 --device cuda:4 &
# python dagma_torch.py --d 40 --seed 2 --device cuda:5 &
# python dagma_torch.py --d 60 --seed 1 --device cuda:6 &
# python dagma_torch.py --d 60 --seed 2 --device cuda:7 &

python test.py --exp_group_idx=v41 --d=40 --v43_method='lasso' --v43_disable_dag_control --device=cuda:6 &
# python test.py --exp_group_idx=v43 --d=40 --v43_method='lasso' --v43_disable_dag_control --device=cuda:6 &
# python test.py --exp_group_idx=v42 --v42_i_idx=2 --v42_ii_idx=1 --d=40 --device=cuda:7 &