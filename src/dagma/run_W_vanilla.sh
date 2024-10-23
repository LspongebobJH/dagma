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

echo "Start fitting NOTEARS from 1 to 10..." > /home/jiahang/dagma/src/dagma/pipe_log.log

cnt=0
ns=(2000)
nodes=(60)
s0_factors=(4 6)
seedsX=( {1..1..1} )

for n in "${ns[@]}"; do
    for d in "${nodes[@]}"; do
        for s0_factor in "${s0_factors[@]}"; do
            for seedX in "${seedsX[@]}"; do

                s0=$(( d * s0_factor ))
                src_note="_SF_normX_sym1"
                dst_note="_SF_normX=sym1"

                mkdir /home/jiahang/dagma/src/dagma/simulated_data/v50/${n}_${d}_${s0}/
                
                python notears_cpu.py \
                    --n=${n} --d=${d} --s0=${s0} \
                    --seed_X=${seedX} \
                    --src_note=${src_note} \
                    --dst_note=${dst_note} \
                    >/home/jiahang/dagma/src/dagma/simulated_data/v50/${n}_${d}_${s0}/log_${seedX}_0${dst_note} 2>&1 &

                wait

                # cnt=$(( cnt + 1 ))
                # _cnt=$(( cnt % 8 ))
                # if [ ${_cnt} -eq 7 ]; then
                #     wait
                # fi
            done
        done
    done
done

echo "End fitting NOTEARS from 1 to 10..." > /home/jiahang/dagma/src/dagma/pipe_log.log