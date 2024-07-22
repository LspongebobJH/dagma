#!/bin/bash

########################################
# Generating W (W_torch)
########################################

# n=2000
# d=20
# s0=80
# version=11
# root_path=simulated_data/v$version

# for i in {1..5}; do
#     python gen_copies.py \
#     --gen_type W_torch \
#     --n $n --s0 $s0 --d $d \
#     --seed_knockoff $i --version $d \
#     --root_path $root_path \
#     --device cuda:6 &
# done

# for i in {6..10}; do
#     python gen_copies.py \
#     --gen_type W_torch \
#     --n $n --s0 $s0 --d $d \
#     --seed_knockoff $i --version $d \
#     --root_path $root_path \
#     --device cuda:7 &
# done

# wait

############################################
# Generating W (W_torch), sweep over n_nodes
############################################

run() {
    data_version=11

    ####################
    # initialize data dir
    ####################
    # ./create_data_dir.sh $data_version $nodes $alpha $src_data_version X
    # ./create_data_dir.sh $data_version $nodes $alpha $src_data_version knockoff

    ####################
    # initialization log
    # dir
    ####################

    direc=logs/log_temp/v${data_version}/
    if [ ! -d "$direc" ]; then
        mkdir $direc
    fi
    
    ####################
    # fitting W
    ####################

    n=2000

    nodes=(200)
    for d in "${nodes[@]}"; do

        cuda_idx=6
        for i in {1..5}; do
            s0=$(( d * 4 ))
            stdbuf -o0 -e0 \
            python gen_copies.py \
            --gen_type W_torch \
            --n $n --s0 $s0 --d $d \
            --seed_knockoff $i \
            --root_path simulated_data/v${data_version} \
            --version ${d} \
            --device cuda:${cuda_idx} > logs/log_temp/v${data_version}/v${d}_${i} 2>&1 &
        done

        cuda_idx=7
        for i in {6..10}; do
            s0=$(( d * 4 ))
            stdbuf -o0 -e0 \
            python gen_copies.py \
            --gen_type W_torch \
            --n $n --s0 $s0 --d $d \
            --seed_knockoff $i \
            --root_path simulated_data/v${data_version} \
            --version ${d} \
            --device cuda:${cuda_idx} > logs/log_temp/v${data_version}/v${d}_${i} 2>&1 &
        done

        # wait
        
    done

}

run
