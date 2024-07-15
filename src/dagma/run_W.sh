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

run() {
    order=$1 # order of deconv if needed
    alpha=$2 # decay of deconv if needed
    data_version=$3
    deconv_type_dagma=$4
    cuda_idx=$5
    T=$6
    # warm_iter=$6
    src_data_version=11
    n=2000
    d=$7
    nodes=$d

    ####################
    # initialize data dir
    ####################
    ./create_data_dir.sh $data_version $nodes $alpha $src_data_version X
    ./create_data_dir.sh $data_version $nodes $alpha $src_data_version knockoff

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

    for i in {1..5}; do
        s0=$(( d * 4 ))
        stdbuf -o0 -e0 \
        python gen_copies.py \
        --gen_type W_torch \
        --n $n --s0 $s0 --d $d \
        --seed_knockoff $i \
        --root_path simulated_data/v${data_version}/v${data_version}_${alpha} \
        --T $T \
        --version $d \
        --device cuda:${cuda_idx} > logs/log_temp/v${data_version}/v${data_version}_${nodes}_${alpha}_${i}_${deconv_type_dagma} 2>&1 &
    done
}

run -1 -1 32 -1 0 6 40
run -1 -1 32 -1 1 6 60
