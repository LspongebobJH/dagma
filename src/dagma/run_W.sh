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
    data_version=$1
    d=$2
    cuda_idx=$3

    dst_data_version=$data_version
    src_data_version=11

    n=2000
    s0=$(( d * 4 ))
    method_list=("elastic" "lasso")

    for method in "${method_list[@]}"; do
        version=${d}_${s0}_1_${method}_disable_dag_control

        ./create_data_dir.sh X $dst_data_version $version $src_data_version ${d}
        
        ####################
        # fitting W
        ####################

        # nodes=(60 80)
        # for d in "${nodes[@]}"; do
        

        for i in {1..1}; do
            stdbuf -o0 -e0 \
            python gen_copies.py \
            --gen_type W_torch \
            --n $n --s0 $s0 --d $d \
            --seed_knockoff $i \
            --root_path simulated_data/v${data_version} \
            --version ${version} \
            --device cuda:${cuda_idx} > logs/log_temp/v${data_version}/v${version} 2>&1 &
        done

        cuda_idx=$(( cuda_idx + 1 ))
    done
}

data_version=43
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

direc=logs/log_temp/v${data_version}/
if [ ! -d "$direc" ]; then
    mkdir $direc
fi


d=40
cuda_idx=4
run $data_version $d $cuda_idx

d=80
cuda_idx=6
run $data_version $d $cuda_idx