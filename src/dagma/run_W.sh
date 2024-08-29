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
    dst_data_version=$1
    d=$2
    cuda_idx=$3

    # src_data_version=11

    n=2000
    s0=$(( d * 6 ))
    norms=(col row)
    # version_list=(${d}_${s0}_option_5_OLS_col ${d}_${s0}_option_5_OLS_row)
    seedsX=(86 85 52 40 18 26 74 63 2 1)

    # for version in "${version_list[@]}"; do
    for norm in "${norms[@]}"; do

        version=${d}_${s0}_option_5_OLS_${norm}
        # ./create_data_dir.sh X $dst_data_version $version $src_data_version ${d}

        target_dir=/home/jiahang/dagma/src/dagma/simulated_data/v${dst_data_version}/v$version/W
        if [ ! -d "$target_dir$" ]; then
            mkdir -p ${target_dir}
        fi
    
        # for (( seedX=1; seedX<=100; seedX++ )); do
        for seedX in "${seedsX[@]}"; do
            stdbuf -o0 -e0 \
            python gen_copies.py \
            --gen_type W_torch \
            --n $n --s0 $s0 --d $d \
            --seed_X $seedX \
            --seed_knockoff 1 \
            --norm $norm \
            --root_path simulated_data/v${dst_data_version} \
            --version ${version} \
            --device cuda:${cuda_idx} > simulated_data/v${dst_data_version}/v${version}/W/log_${seedX}_1_0 2>&1 &

            cuda_idx=$(( cuda_idx + 1 ))
            cuda_idx=$(( cuda_idx % 8))

            # _seedX=$(( seedX % 20 ))
            # if [ ${_seedX} -eq 10 ]; then
            #     wait
            # fi
        done
        wait
    done
    
}

data_version=46
# src_data_version=11
# dst_data_version=${data_version}

####################
# initialize data dir
####################
# ./create_data_dir.sh X $dst_data_version ${d} $src_data_version ${d}
# ./create_data_dir.sh knockoff $dst_data_version ${d} $src_data_version ${d}

####################
# initialization log
# dir
####################

# direc=logs/log_temp/v${data_version}/
# if [ ! -d "$direc" ]; then
#     mkdir $direc
# fi

# d=20
# cuda_idx=0
# run $data_version $d $cuda_idx

# wait

# d=40
# cuda_idx=0
# run $data_version $d $cuda_idx

# wait

# d=20
# cuda_idx=3
# run $data_version $d $cuda_idx

# wait

# d=40
# cuda_idx=0
# run $data_version $d $cuda_idx

# wait

d=60
cuda_idx=0
run $data_version $d $cuda_idx

# wait

# d=80
# cuda_idx=0
# run $data_version $d $cuda_idx

# wait

# d=100
# cuda_idx=0
# run $data_version $d $cuda_idx

# d=60
# cuda_idx=5
# run $data_version $d $cuda_idx

# d=100
# cuda_idx=6
# run $data_version $d $cuda_idx
