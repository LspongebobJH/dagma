#!/bin/bash

########################################
# Generating W (W_torch)
########################################

# for i in {1..5}; do
#     python gen_copies.py \
#     --gen_type W_torch \
#     --n $n --s0 $s0 --d $d \
#     --seed_knockoff $i --version $version \
#     --device cuda:6 &
# done

# for i in {6..10}; do
#     python gen_copies.py \
#     --gen_type W_torch \
#     --n $n --s0 $s0 --d $d \
#     --seed_knockoff $i --version $version \
#     --device cuda:7 &
# done

# wait

############################################
# Generating W (W_torch), sweep over n_nodes
############################################

n=2000
nodes=(40 60 80)
for d in "${nodes[@]}"; do
    for i in {1..5}; do
        s0=$(( d * 4 ))
        python gen_copies.py \
        --gen_type W_torch \
        --n $n --s0 $s0 --d $d \
        --seed_knockoff $i \
        --root_path simulated_data/v12 \
        --version $d \
        --device cuda:6 &
    done

    for i in {6..10}; do
        s0=$(( d * 4 ))
        python gen_copies.py \
        --gen_type W_torch \
        --n $n --s0 $s0 --d $d \
        --seed_knockoff $i \
        --root_path simulated_data/v12 \
        --version $d \
        --device cuda:7 &
    done
    wait
done