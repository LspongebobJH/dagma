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

n=2000
nodes=(20 40 60 80)
data_version=33
for d in "${nodes[@]}"; do
    for i in {1..10}; do
        s0=$(( d * 5 ))
        python gen_copies.py --gen_type knockoff \
        --n $n --d $d --s0 $s0 --seed_knockoff $i \
        --root_path simulated_data/v${data_version} \
        --version ${d}_${s0} &
        if [ $i -eq 10 ]; then
            pid=$!
        fi
    done

    for i in {1..10}; do
        s0=$(( d * 6 ))
        python gen_copies.py --gen_type knockoff \
        --n $n --d $d --s0 $s0 --seed_knockoff $i \
        --root_path simulated_data/v${data_version} \
        --version ${d}_${s0} &
        if [ $i -eq 10 ]; then
            pid=$!
        fi
    done

    wait

done