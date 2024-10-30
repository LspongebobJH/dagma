#!/bin/bash

#########################################
# vanilla dagma, dagma_torch.py
#########################################

# d=100
# seedsX=( {1..3..1} )
# choices=(disable_dag disable_dag+l1+l2)
# cuda_idx=5
# cnt=0

# s0=$(( d * 6 ))
# for seedX in "${seedsX[@]}"; do
#     for choice in "${choices[@]}"; do
        
#         src_note="_normX_sym1"
#         dst_note="_normX_sym1_${choice}"

#         if [ $choice = "disable_l1" ]; then
#             python dagma_torch.py --d $d --s0 $s0 \
#             --seed_X $seedX --src_note $src_note --dst_note $dst_note \
#             --disable_l1 \
#             --device cuda:${cuda_idx} \
#             >/home/jiahang/dagma/src/dagma/simulated_data/v39/${d}_${s0}/log_${seedX}_0_${dst_note} 2>&1 &
#         elif [ $choice = "disable_l2" ]; then
#             python dagma_torch.py --d $d --s0 $s0 \
#             --seed_X $seedX --src_note $src_note --dst_note $dst_note \
#             --disable_l2 \
#             --device cuda:${cuda_idx} \
#             >/home/jiahang/dagma/src/dagma/simulated_data/v39/${d}_${s0}/log_${seedX}_0_${dst_note} 2>&1 &
#         elif [ $choice = "disable_l1+l2" ]; then
#             python dagma_torch.py --d $d --s0 $s0 \
#             --seed_X $seedX --src_note $src_note --dst_note $dst_note \
#             --disable_l1 --disable_l2 \
#             --device cuda:${cuda_idx} \
#             >/home/jiahang/dagma/src/dagma/simulated_data/v39/${d}_${s0}/log_${seedX}_0_${dst_note} 2>&1 &
#         elif [ $choice = "disable_none" ]; then
#             python dagma_torch.py --d $d --s0 $s0 \
#             --seed_X $seedX --src_note $src_note --dst_note $dst_note \
#             --device cuda:${cuda_idx} \
#             >/home/jiahang/dagma/src/dagma/simulated_data/v39/${d}_${s0}/log_${seedX}_0_${dst_note} 2>&1 &

#         elif [ $choice = "disable_dag" ]; then
#             python dagma_torch.py --d $d --s0 $s0 \
#             --seed_X $seedX --src_note $src_note --dst_note $dst_note \
#             --device cuda:${cuda_idx} \
#             --disable_dag \
#             >/home/jiahang/dagma/src/dagma/simulated_data/v39/${d}_${s0}/log_${seedX}_0_${dst_note} 2>&1 &
#         elif [ $choice = "disable_dag+l1+l2" ]; then
#             python dagma_torch.py --d $d --s0 $s0 \
#             --seed_X $seedX --src_note $src_note --dst_note $dst_note \
#             --device cuda:${cuda_idx} \
#             --disable_dag --disable_l1 --disable_l2 \
#             >/home/jiahang/dagma/src/dagma/simulated_data/v39/${d}_${s0}/log_${seedX}_0_${dst_note} 2>&1 &
#         fi

#         cuda_idx=$(( cuda_idx + 1 ))
#         if [ $cuda_idx -eq 8 ]; then
#             cuda_idx=5
#         fi

#         cnt=$(( cnt + 1 ))
#         _cnt=$(( cnt % 30 ))
        
#         if [ ${_cnt} -eq 29 ]; then
#             wait
#         fi
#     done
# done



#########################################
# vanilla GENIE3, genie3.py
#########################################

# echo "Start fitting grnboost2 from 1 to 10..." > /home/jiahang/dagma/src/dagma/pipe_log.log

# n_nodes=100
# n_edges=600

# # ntrees_list=(5000 500)
# # max_feat_list=(0.1 0.9)
# # max_sample_list=(0.9 0.1)
# seedsX=( {1..3..1} )
# model_types=(OLS L1 L2 L1+L2)

# for seedX in "${seedsX[@]}"; do
#     for model_type in "${model_types[@]}"; do
    
#         src_note="_normX_sym1"
#         dst_note="_normX=sym1_${model_type}"
#         python genie3.py \
#             --d=${n_nodes} --s0=${n_edges} --seed_X=${seedX} \
#             --src_note=${src_note} \
#             --dst_note=${dst_note} \
#             --force_save \
#             --model_type=${model_type} \
#             --nthreads=2 \
#             >/Users/jiahang/Documents/dagma/src/dagma/simulated_data/v48/${n_nodes}_${n_edges}/log_${seedX}_0_${dst_note} 2>&1 &
#             # >/home/jiahang/dagma/src/dagma/simulated_data/v48/${n_nodes}_${n_edges}/log_${seedX}_0_${dst_note} 2>&1 &
#             # --nthreads=4 --use_grnboost2 \

        
#         # _cnt=$(( seedsX % 10 ))
#         # if [ ${_cnt} -eq 9 ]; then
#         #     wait
#         # fi
#     done
#     wait
# done

# echo "End fitting grnboost2 from 1 to 10..." >> /home/jiahang/dagma/src/dagma/pipe_log.log

# seedsX=( {11..30..1} )
# for seedX in "${seedsX[@]}"; do
#     python genie3.py --d=100 --s0=600 --seed_X=${seedX} --note="_normX_sym1" --nthreads=4 &
#     _cnt=$(( seedsX % 10 ))
#     if [ ${_cnt} -eq 9 ]; then
#         wait
#     fi
# done

#########################################
# vanilla NOTEARS, notears_cpu.py
#########################################

# echo "Start fitting NOTEARS from 1 to 10..." > /home/jiahang/dagma/src/dagma/pipe_log.log

# cnt=0
# fit_W_version=50
# ns=(2000)
# nodes=(40 60)
# s0_factors=(4 6)
# seedsX=( {1..10..1} )

# for n in "${ns[@]}"; do
#     for d in "${nodes[@]}"; do
#         for s0_factor in "${s0_factors[@]}"; do
#             for seedX in "${seedsX[@]}"; do

#                 s0=$(( d * s0_factor ))
#                 src_note="_normX_sym1"
#                 dst_note="_normX=sym1"

#                 mkdir /home/jiahang/dagma/src/dagma/simulated_data/v${fit_W_version}/${n}_${d}_${s0}/
                
#                 python notears_cpu.py \
#                     --n=${n} --d=${d} --s0=${s0} \
#                     --seed_X=${seedX} \
#                     --src_note=${src_note} \
#                     --dst_note=${dst_note} \
#                     >/home/jiahang/dagma/src/dagma/simulated_data/v${fit_W_version}/${n}_${d}_${s0}/log_${seedX}_0${dst_note} 2>&1 &

#                 # wait

#                 cnt=$(( cnt + 1 ))
#                 _cnt=$(( cnt % 16 ))
#                 if [ ${_cnt} -eq 15 ]; then
#                     wait
#                 fi
#             done
#         done
#     done
# done

# echo "End fitting NOTEARS from 1 to 10..." > /home/jiahang/dagma/src/dagma/pipe_log.log

#########################################
# vanilla golem, golem.py
#########################################

echo "Start fitting NOTEARS from 1 to 10..." > /home/jiahang/dagma/src/dagma/pipe_log.log

cnt=0
# fit_W_version=52 # golem
# fit_W_version=54 # dag_gnn
# fit_W_version=(52 54)
models=(golem dag_gnn)
lambda_l1s=(0.1 0.5 0.9)
lambda_l2s=(0.1 0.5 0.9)
ns=(2000)
nodes=(40)
s0_factors=(6)
seedsX=(1)

for lambda_l1 in "${lambda_l1s[@]}"; do
    for lambda_l2 in "${lambda_l2s[@]}"; do
        for model in "${models[@]}"; do
            for n in "${ns[@]}"; do
                for d in "${nodes[@]}"; do
                    for s0_factor in "${s0_factors[@]}"; do
                        for seedX in "${seedsX[@]}"; do

                            s0=$(( d * s0_factor ))
                            X_name="${n}_${d}_${s0}_normX_sym1"
                            dst_note="_normX=sym1_l1=${lambda_l1}_l2=${lambda_l2}"
                            dst_name="W_${seedX}_0${dst_note}"

                            if [ $model = 'golem' ]; then
                                fit_W_version=52
                            elif [ $model = 'dag_gnn' ]; then
                                fit_W_version=54
                            fi

                            target_dir=/home/jiahang/dagma/src/dagma/simulated_data/v${fit_W_version}/${n}_${d}_${s0}/
                            if [ ! -d "$target_dir$" ]; then
                                mkdir -p ${target_dir}
                            fi
                            
                            python fit_W.py \
                                --n=${n} --d=${d} --s0=${s0} \
                                --seed_X=${seedX} \
                                --model=${model} \
                                --lambda_l1=${lambda_l1} \
                                --lambda_l2=${lambda_l2} \
                                --X_name=${X_name} \
                                --dst_version=${fit_W_version} \
                                --dst_name=${dst_name} \
                                >/home/jiahang/dagma/src/dagma/simulated_data/v${fit_W_version}/${n}_${d}_${s0}/log_${seedX}_0${dst_note} 2>&1 &

                            # wait

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

wait

echo "End fitting NOTEARS from 1 to 10..." > /home/jiahang/dagma/src/dagma/pipe_log.log
