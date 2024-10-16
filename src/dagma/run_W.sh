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

    n=2000
    s0=$(( d * 6 ))
    suffixs=(_normX=sym1_option=5_knock=PLS_model=L1+L2
            )
    seedsX=( {1..3..1} )
    seedsKnockoff=( 1 )
    seedsW=( 0 )
    alpha_list=(5 10 20)
    l1_ratio_list=(0.1 0.5 0.9)
    cnt=0


    for suffix in "${suffixs[@]}"; do
        for alpha in "${alpha_list[@]}"; do
            for l1_ratio in "${l1_ratio_list[@]}"; do
                # version=${d}_${s0}${suffix}
                version=${d}_${s0}${suffix}_alpha=${alpha}_l1_ratio=${l1_ratio}

                target_dir=/home/jiahang/dagma/src/dagma/simulated_data/v${dst_data_version}/v$version/W
                if [ ! -d "$target_dir$" ]; then
                    mkdir -p ${target_dir}
                fi

                # for (( seedX=1; seedX<=100; seedX++ )); do
                for seedKnockoff in "${seedsKnockoff[@]}"; do
                    for seedX in "${seedsX[@]}"; do
                        for seedW in "${seedsW[@]}"; do
                            stdbuf -o0 -e0 \
                            python gen_copies.py \
                            --gen_type W_L1+L2 \
                            --elastic_alpha=${alpha} --elastic_l1_ratio=${l1_ratio} \
                            --n $n --s0 $s0 --d $d \
                            --seed_X $seedX \
                            --seed_knockoff $seedKnockoff \
                            --seed_model $seedW \
                            --root_path simulated_data/v${dst_data_version} \
                            --version ${version} \
                            --device cuda:${cuda_idx} > simulated_data/v${dst_data_version}/v${version}/W/log_${seedX}_${seedKnockoff}_${seedW} 2>&1 &

                            cuda_idx=$(( cuda_idx + 1))
                            if [ $cuda_idx -eq 8 ]; then
                                cuda_idx=5
                            fi

                            cnt=$(( cnt + 1 ))
                            _cnt=$(( cnt % 20 ))
                            
                            if [ ${_cnt} -eq 19 ]; then
                                wait
                            fi

                            # _seedX=$(( seedX % 20 ))
                            # if [ ${_seedX} -eq 10 ]; then
                            #     wait
                            # fi
                        done
                    done
                done
            done
        done
    done
    
}

data_version=49
# data_version=47


# d=20
# cuda_idx=5
# run $data_version $d $cuda_idx

# # wait

# d=40
# cuda_idx=6
# run $data_version $d $cuda_idx

# # wait

# d=60
# cuda_idx=7
# run $data_version $d $cuda_idx

# wait

# d=80
# cuda_idx=5
# run $data_version $d $cuda_idx

echo "Start fitting W of [X, X'] from 1 to 3..." >> /home/jiahang/dagma/src/dagma/pipe_log.log

d=100
cuda_idx=5
run $data_version $d $cuda_idx

wait

echo "End fitting W of [X, X'] from 1 to 3..." >> /home/jiahang/dagma/src/dagma/pipe_log.log
