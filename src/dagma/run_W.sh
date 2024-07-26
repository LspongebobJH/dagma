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

    n=2000
    s0=$(( d * 4 ))
    version=${d}
    
    ####################
    # fitting W
    ####################

    # nodes=(60 80)
    # for d in "${nodes[@]}"; do
    

    for i in {1..10}; do
        stdbuf -o0 -e0 \
        python gen_copies.py \
        --gen_type W_torch \
        --n $n --s0 $s0 --d $d \
        --seed_knockoff $i \
        --root_path simulated_data/v${data_version} \
        --version ${version} \
    --device cuda:${cuda_idx} > logs/log_temp/v${data_version}/v${version}_${i} 2>&1 &
    done

        # wait
        
    # done

}

data_version=34


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

d=80
cuda_idx=4
run $data_version $d $cuda_idx

d=100
cuda_idx=5
run $data_version $d $cuda_idx

# d=120
# cuda_idx=7
# run $data_version $d $cuda_idx

# d=140
# cuda_idx=8
# run $data_version $d $cuda_idx
