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

echo "Start generating X from 1 to 10..." > /home/jiahang/dagma/src/dagma/pipe_log.log

n=2000
d=100
s0=600

d1_list=(10 40 80)
seedsX=( {1..3..1} )
for d1 in "${d1_list[@]}"; do
    for seedX in "${seedsX[@]}"; do
        # s0=$(( d * 2 ))
        
        d2=$(( d - d1 ))
        
        python gen_copies.py --gen_type X_bi \
        --n $n --d $d --d1 $d1 --d2 $d2 --s0 $s0 \
        --norm_data_gen sym_1 \
        --seed_X $seedX \
        --root_path simulated_data/v11 \
        --version ${d}_${s0}_bipartite_${d1}_${d2}_normX_sym1 &

        # _seedX=$(( seedX % 20 ))

        # if [ ${_seedX} -eq 19 ]; then
        #     wait
        # fi
    done
done

echo "End generating X from 1 to 10." >> /home/jiahang/dagma/src/dagma/pipe_log.log

wait

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

# echo "Start generating knockoff from 1 to 3..." > /home/jiahang/dagma/src/dagma/pipe_log.log

# data_option=X
# dst_data_version=49
# src_data_version=11
# # fit_W_version=39
# fit_W_version=48
# n=2000
# cuda_idx=5
# nodes=(100)
# seedsX=( {1..3..1} )
# seedsKnockoff=( 1 )
# suffixs=(_option_5_PLS_disable_norm_grnboost2
#          _option_5_OLS_disable_norm_grnboost2
#          )
# cnt=0

# for d in "${nodes[@]}"; do
#     for suffix in "${suffixs[@]}"; do
#         for seedKnockoff in "${seedsKnockoff[@]}"; do
#             s0=$(( d * 6 ))
            
#             dst_version=${d}_${s0}${suffix}
#             src_version=${d}_${s0}
            
#             ./create_data_dir.sh $data_option $dst_data_version $dst_version $src_data_version $src_version

#             target_dir=/home/jiahang/dagma/src/dagma/simulated_data/v${dst_data_version}/v$dst_version/knockoff
#             if [ ! -d "$target_dir$" ]; then
#                 mkdir -p ${target_dir}
#             fi

#             for seedX in "${seedsX[@]}"; do
#                 # PLS
#                 if [ $suffix = '_option_5_PLS_disable_norm_grnboost2' ]; then
#                     CUDA_VISIBLE_DEVICES=${cuda_idx} \
#                     python knockoff_v44.py \
#                     --W_type=W_est \
#                     --data_version=v${dst_data_version} \
#                     --dst_version=v${dst_version} \
#                     --fit_W_version=v${fit_W_version} \
#                     --option=5 \
#                     --d=${d} --s0=${s0} \
#                     --method_diagn_gen=PLS \
#                     --dedup \
#                     --device=cuda:${cuda_idx} \
#                     --seed_X=${seedX} \
#                     --seed_knockoff=${seedKnockoff} \
#                     >/home/jiahang/dagma/src/dagma/simulated_data/v${dst_data_version}/v$dst_version/knockoff/log_${seedX}_${seedKnockoff} 2>&1 &

#                     # --note="_grnboost2" \

#                 elif [ $suffix = '_option_5_OLS_disable_norm_grnboost2' ]; then
#                     CUDA_VISIBLE_DEVICES=${cuda_idx} \
#                     python knockoff_v44.py \
#                     --W_type=W_est \
#                     --data_version=v${dst_data_version} \
#                     --dst_version=v${dst_version} \
#                     --fit_W_version=v${fit_W_version} \
#                     --option=5 \
#                     --d=${d} --s0=${s0} \
#                     --method_diagn_gen=OLS \
#                     --dedup \
#                     --device=cuda:${cuda_idx} \
#                     --seed_X=${seedX} \
#                     --seed_knockoff=${seedKnockoff} \
#                     >/home/jiahang/dagma/src/dagma/simulated_data/v${dst_data_version}/v$dst_version/knockoff/log_${seedX}_${seedKnockoff} 2>&1 &

#                     # --note="_grnboost2" \

#                 # control cuda device
#                 cuda_idx=$(( cuda_idx + 1))
#                 if [ $cuda_idx -eq 8 ]; then
#                     cuda_idx=5
#                 fi
                
#                 # control parallel process number
#                 cnt=$(( cnt + 1 ))
#                 _cnt=$(( cnt % 20 ))
                
#                 if [ ${_cnt} -eq 19 ]; then
#                     wait
#                 fi
#             done
#         done
#     done
# done

# wait

# echo "End generating knockoff from 1 to 3." > /home/jiahang/dagma/src/dagma/pipe_log.log

#########################################
# vanilla dagma, dagma_torch.py
#########################################

# nodes=(20 40 60 80 100)
# # notes=(_normX_B1Col _normX_sym1)
# # seedsX=(86 85 52 40 18 26 74 63 2 1)
# seedsX=( {11..30..1} )
# cuda_idx=5
# cnt=0
# for d in "${nodes[@]}"; do
#     s0=$(( d * 6 ))
#     # for note in "${notes[@]}"; do
#         for seedX in "${seedsX[@]}"; do
#         # for (( seedX=1; seedX<=5; seedX++ )); do
#             python dagma_torch.py --d $d --s0 $s0 \
#             --seed_X $seedX --note "_normX_sym1" \
#             --device cuda:${cuda_idx} &

#             cuda_idx=$(( cuda_idx + 1 ))
#             if [ $cuda_idx -eq 8 ]; then
#                 cuda_idx=5
#             fi

#             cnt=$(( cnt + 1 ))
#             _cnt=$(( cnt % 30 ))
            
#             if [ ${_cnt} -eq 29 ]; then
#                 wait
#             fi
#         done
#     # done
# done

#########################################
# vanilla GENIE3, genie3.py
#########################################

echo "Start fitting grnboost2 from 1 to 10..." >> /home/jiahang/dagma/src/dagma/pipe_log.log

d1_list=(10 40 80)
n_nodes=100
n_edges=600

seedsX=( {1..3..1} )
for d1 in "${d1_list[@]}"; do
    for seedX in "${seedsX[@]}"; do
        d2=$(( n_nodes - d1 ))
        src_note="_bipartite_${d1}_${d2}_normX_sym1"
        dst_note="_bipartite_${d1}_${d2}_normX_sym1_disable_norm_grnboost2"

        python genie3.py \
            --d=${n_nodes} --d1=$d1 --d2=$d2 --s0=${n_edges} --seed_X=${seedX} \
            --src_note=${src_note} \
            --dst_note=${dst_note} \
            --disable_norm \
            --force_save \
            --nthreads=4 --use_grnboost2 \
            >/home/jiahang/dagma/src/dagma/simulated_data/v48/${n_nodes}_${n_edges}/log_${seedX}_0_${dst_note} 2>&1 &
        # _cnt=$(( seedsX % 10 ))
        # if [ ${_cnt} -eq 9 ]; then
        #     wait
        # fi
    done
done

echo "End fitting grnboost2 from 1 to 10..." >> /home/jiahang/dagma/src/dagma/pipe_log.log

# seedsX=( {11..30..1} )
# for seedX in "${seedsX[@]}"; do
#     python genie3.py --d=100 --s0=600 --seed_X=${seedX} --note="_normX_sym1" --nthreads=4 &
#     _cnt=$(( seedsX % 10 ))
#     if [ ${_cnt} -eq 9 ]; then
#         wait
#     fi
# done

#########################################
# Misc
#########################################

# python test.py --exp_group_idx=v43 --d=20 --v43_method='elastic' --v43_disable_dag_control --device=cuda:6 &
# python test.py --exp_group_idx=v43 --d=100 --v43_method='elastic' --v43_disable_dag_control --device=cuda:7 &
# python test.py --exp_group_idx=v44 --v44_option=1 --lasso_alpha=OLS --W_type=W_true \
#     --d=20 --device=cuda:5 &




