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
# nodes=(20 40 60 80 100)
# for d in "${nodes[@]}"; do
#     s0=$(( d * 6 ))
#     python gen_copies.py --gen_type X \
#     --n $n --d $d --s0 $s0 \
#     --root_path simulated_data/v11 \
#     --version ${d}_${s0} &
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
dst_data_version=44
src_data_version=11
n=2000
cuda_idx=0
nodes=(20 40 60 80 100)
options=(5)

for option in "${options[@]}"; do
    suffix=_option_${option}_lasso_OLS
    for d in "${nodes[@]}"; do
        s0=$(( d * 6 ))
        dst_version=${d}_${s0}${suffix}
        src_version=${d}_${s0}
        
        ./create_data_dir.sh $data_option $dst_data_version $dst_version $src_data_version $src_version

        target_dir=/home/jiahang/dagma/src/dagma/simulated_data/v${dst_data_version}/v$dst_version/knockoff
        if [ ! -d "$target_dir$" ]; then
            mkdir -p ${target_dir}
        fi

        for i in {1..10}; do
            python new_knockoff_generation.py \
            --exp_group_idx=v44 --v44_option=${option} \
            --d=${d} --s0=${s0} --W_type=W_est \
            --method_diagn_gen=lasso --lasso_alpha=OLS \
            --seed_knockoff=$i --notes=$suffix \
            >/home/jiahang/dagma/src/dagma/simulated_data/v${dst_data_version}/v$dst_version/knockoff/log_${i} 2>&1 &

            # python gen_copies.py --gen_type knockoff --knock_type knockoff_diagn --n $n --d $d --s0 $s0 --seed_knockoff $i \
            # --method_diagn_gen=lasso --lasso_alpha=OLS \
            # --root_path simulated_data/v${dst_data_version} \
            # --version $dst_version --device cuda:${cuda_idx} \
            # >/home/jiahang/dagma/src/dagma/simulated_data/v${dst_data_version}/v$dst_version/knockoff/log_${i} 2>&1 &
        done
        # cuda_idx=$(( cuda_idx + 1 ))
    done
    # wait
done

#########################################
# Misc
#########################################

# python dagma_torch.py --d 10 --seed 1 --device cuda:0 &
# python dagma_torch.py --d 10 --seed 2 --device cuda:1 &
# python dagma_torch.py --d 20 --seed 1 --device cuda:2 &
# python dagma_torch.py --d 20 --seed 2 --device cuda:3 &
# python dagma_torch.py --d 40 --seed 1 --device cuda:4 &
# python dagma_torch.py --d 40 --seed 2 --device cuda:5 &
# python dagma_torch.py --d 60 --seed 1 --device cuda:6 &
# python dagma_torch.py --d 60 --seed 2 --device cuda:7 &

# python test.py --exp_group_idx=v43 --d=20 --v43_method='elastic' --v43_disable_dag_control --device=cuda:6 &
# python test.py --exp_group_idx=v43 --d=100 --v43_method='elastic' --v43_disable_dag_control --device=cuda:7 &
# python test.py --exp_group_idx=v44 --v44_option=1 --lasso_alpha=OLS --W_type=W_true \
#     --d=20 --device=cuda:5 &




