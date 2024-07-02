#!/bin/bash

# version=7
# s0=1000
# d=60
# n=2000

########################################
# Generating X
########################################

# python gen_copies.py --gen_type X --n $n --s0 $s0 --d $d --version $version

########################################
# Generating X, sweep over n_nodes
########################################

# n=2000
# nodes=(10 40 60 80 100 200 400)
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
#     python gen_copies.py --gen_type knockoff --n $n --s0 $s0 --d $d  --seed_knockoff $i --version $version &
# done

# wait

#########################################
# Generating Knockoff, sweep over n_nodes
#########################################

n=2000
nodes=(40 60 80)
for d in "${nodes[@]}"; do
    for i in {1..10}; do
        s0=$(( d * 4 ))
        python gen_copies.py --gen_type knockoff \
        --n $n --d $d --s0 $s0 --seed_knockoff $i \
        --root_path simulated_data/v12 \
        --niter 5000 \
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