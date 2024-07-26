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

data_option=X
dst_data_version=34
# dst_final_version=20
src_data_version=11
# src_final_version=20
n=2000
nodes=(80 100 120 140 160 180 200)
for d in "${nodes[@]}"; do
    
    s0=$(( d * 4 ))
    ./create_data_dir.sh $data_option $dst_data_version ${d} $src_data_version ${d}

    for i in {1..5}; do
        python gen_copies.py --gen_type knockoff --knock_type knockoff_diagn \
        --n $n --d $d --s0 $s0 --seed_knockoff $i \
        --root_path simulated_data/v${dst_data_version} \
        --version ${d} --device cuda:6 &
    done

    for i in {6..10}; do
        python gen_copies.py --gen_type knockoff --knock_type knockoff_diagn \
        --n $n --d $d --s0 $s0 --seed_knockoff $i \
        --root_path simulated_data/v${dst_data_version} \
        --version ${d} --device cuda:7 &
    done

    wait

done