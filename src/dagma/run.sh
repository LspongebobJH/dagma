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
# nodes=(120 140 160 180 200)
# for d in "${nodes[@]}"; do
#     s0=$(( d * 4 ))
#     python gen_copies.py --gen_type X \
#     --n $n --d $d --s0 $s0 \
#     --root_path simulated_data/v11 \
#     --version $d &
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
nodes=(120 140 160 180 200)
data_version=11
for d in "${nodes[@]}"; do
    for i in {1..10}; do
        s0=$(( d * 4 ))
        python gen_copies.py --gen_type knockoff \
        --n $n --d $d --s0 $s0 --seed_knockoff $i \
        --root_path simulated_data/v${data_version} \
        --version $d &
        if [ $i -eq 10 ]; then
            pid=$!
        fi
    done

    wait -n $pid
    if [ $? -ne 0 ]; then
        break
    fi
    wait
done