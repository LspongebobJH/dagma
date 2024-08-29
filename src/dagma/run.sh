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
# Generating X, sweep over n_nodes or 
# seeds
########################################

# n=2000
# nodes=(60)
# for d in "${nodes[@]}"; do
#     for (( seed=21; seed<=100; seed++ )); do
#         s0=$(( d * 6 ))
#         python gen_copies.py --gen_type X \
#         --n $n --d $d --s0 $s0 \
#         --seed_X $seed \
#         --root_path simulated_data/v11 \
#         --version ${d}_${s0} &
#     done
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

data_option=X
dst_data_version=46
src_data_version=11
n=2000
cuda_idx=0
nodes=(60)
seedsX=(86 85 52 40 18 26 74 63 2 1)
norms=(row)

for d in "${nodes[@]}"; do
    for norm in "${norms[@]}"; do
        option=5
        suffix=_option_${option}_OLS_${norm}
        s0=$(( d * 6 ))

        dst_version=${d}_${s0}${suffix}
        src_version=${d}_${s0}
        
        ./create_data_dir.sh $data_option $dst_data_version $dst_version $src_data_version $src_version

        target_dir=/home/jiahang/dagma/src/dagma/simulated_data/v${dst_data_version}/v$dst_version/knockoff
        if [ ! -d "$target_dir$" ]; then
            mkdir -p ${target_dir}
        fi

        # for (( seedX=1; seedX<=100; seedX++ )); do
        for seedX in "${seedsX[@]}"; do
            CUDA_VISIBLE_DEVICES=${cuda_idx} \
            python knockoff_v44.py \
            --data_version=v${dst_data_version} \
            --option=${option} \
            --d=${d} --s0=${s0} \
            --method_diagn_gen=OLS_cuda \
            --device=cuda:${cuda_idx} \
            --seed_X=${seedX} \
            --seed_knockoff=1 \
            --norm=${norm} \
            --notes=$suffix \
            >/home/jiahang/dagma/src/dagma/simulated_data/v${dst_data_version}/v$dst_version/knockoff/log_${seedX}_1 2>&1 &

            cuda_idx=$(( cuda_idx + 1 ))
            cuda_idx=$(( cuda_idx % 8 ))

            # _seedX=$(( seedX % 20 ))

            # if [ ${_seedX} -eq 19 ]; then
            #     wait
            # fi
        
        done
    done
    # wait
done

#########################################
# Misc
#########################################

# nodes=(60 80 20 40 100)
# seeds=(1 2 3)
# cuda_idx=0
# for d in "${nodes[@]}"; do
#     s0=$(( d * 6 ))
#     for seed in "${seeds[@]}"; do
#         python dagma_torch.py --d $d --s0 $s0 --seed $seed --device cuda:${cuda_idx} &
#         cuda_idx=$(( cuda_idx + 1 ))
#         cuda_idx=$(( cuda_idx % 8 ))
#     done
# done

# python test.py --exp_group_idx=v43 --d=20 --v43_method='elastic' --v43_disable_dag_control --device=cuda:6 &
# python test.py --exp_group_idx=v43 --d=100 --v43_method='elastic' --v43_disable_dag_control --device=cuda:7 &
# python test.py --exp_group_idx=v44 --v44_option=1 --lasso_alpha=OLS --W_type=W_true \
#     --d=20 --device=cuda:5 &




