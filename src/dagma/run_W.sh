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
    n=$2
    d=$3
    s0_factor=$4
    cuda_idx=$5

    s0=$(( d * s0_factor ))
    suffixs=(_normX=sym1_option=5_knock=PLS
            )
    lambda_l1s=(0.1 0.5 0.9)
    lambda_l2s=(0.1 0.5 0.9)
    seedsX=( 1 )
    seedsKnockoff=( 1 )
    seedsW=( 0 )
    cnt=0

    models=(golem dag-gnn)

    for model in "${models[@]}"; do
        if [ $model = 'golem' ]; then
            dst_data_version=53
        elif [ $model = 'dag-gnn' ]; then
            dst_data_version=55
        fi
        for lambda_l1 in "${lambda_l1s[@]}"; do
            for lambda_l2 in "${lambda_l2s[@]}"; do
                for suffix in "${suffixs[@]}"; do
                    # version=${d}_${s0}${suffix}
                    version=${n}_${d}_${s0}${suffix}_l1=${lambda_l1}_l2=${lambda_l2}

                    target_dir=/home/jiahang/dagma/src/dagma/simulated_data/v${dst_data_version}/v$version/W
                    if [ ! -d "$target_dir$" ]; then
                        mkdir -p ${target_dir}
                    fi

                    for seedKnockoff in "${seedsKnockoff[@]}"; do
                        for seedX in "${seedsX[@]}"; do
                            for seedW in "${seedsW[@]}"; do
                                stdbuf -o0 -e0 \
                                python gen_copies.py \
                                --gen_type W_${model} \
                                --lambda_l1 ${lambda_l1} \
                                --lambda_l2 ${lambda_l2} \
                                --n $n --s0 $s0 --d $d \
                                --seed_X $seedX \
                                --seed_knockoff $seedKnockoff \
                                --seed_model $seedW \
                                --force_save \
                                --root_path simulated_data/v${dst_data_version} \
                                --version ${version} \
                                --device cuda:${cuda_idx} > simulated_data/v${dst_data_version}/v${version}/W/log_${seedX}_${seedKnockoff}_${seedW} 2>&1 &

                                cuda_idx=$(( cuda_idx + 1))
                                if [ $cuda_idx -eq 7 ]; then
                                    cuda_idx=5
                                fi

                                cnt=$(( cnt + 1 ))
                                _cnt=$(( cnt % 20 ))
                                
                                if [ ${_cnt} -eq 19 ]; then
                                    wait
                                fi

                            done
                        done
                    done
                done
            done
        done
    done
    
}
# data_version=55 # dag_gnn
# data_version=53 # golem
# data_version=51 # notears
# data_version=49 # tree
# data_version=47 # dagma
data_version=0

echo "Start fitting W of [X, X'] from 1 to 3..." >> /home/jiahang/dagma/src/dagma/pipe_log.log

n=2000
nodes=(40)
s0_factors=(6)

for d in "${nodes[@]}"; do
    for s0_factor in "${s0_factors[@]}"; do
        cuda_idx=5
        run $data_version $n $d $s0_factor $cuda_idx
        # wait
    done
done

echo "End fitting W of [X, X'] from 1 to 3..." >> /home/jiahang/dagma/src/dagma/pipe_log.log
