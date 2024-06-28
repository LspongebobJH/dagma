#!/bin/bash

# version=7
# s0=1000
# d=60
# n=2000

########################################
# Generating X
########################################

# python gen_copies.py --gen_type X --n $n --s0 $s0 --d $d --version $version

# n=2000
# nodes=(10 40 60 80 100 200 400)
# for d in "${nodes[@]}"; do
#     s0=$(( d * 4 ))
#     python gen_copies.py --gen_type X \
#     --n $n --d $d --s0 $s0 \
#     --root_path simulated_data/v11 \
#     --version $d &
# done
    

########################################
# Generating Knockoff
########################################

# for i in {1..10}; do
#     python gen_copies.py --gen_type knockoff --n $n --s0 $s0 --d $d  --seed_knockoff $i --version $version &
# done

# wait

# n=2000
# nodes=(10 40 60 80 100 200 400)
# for d in "${nodes[@]}"; do
#     for i in {1..10}; do
#         s0=$(( d * 4 ))
#         python gen_copies.py --gen_type knockoff \
#         --n $n --d $d --s0 $s0 --seed_knockoff $i \
#         --root_path simulated_data/v11 \
#         --version $d &
#     done
#     wait
# done

########################################
# Generating W (W_torch)
########################################

# for i in {1..5}; do
#     python gen_copies.py \
#     --gen_type W_torch \
#     --n $n --s0 $s0 --d $d \
#     --seed_knockoff $i --version $version \
#     --device cuda:6 &
# done

# for i in {6..10}; do
#     python gen_copies.py \
#     --gen_type W_torch \
#     --n $n --s0 $s0 --d $d \
#     --seed_knockoff $i --version $version \
#     --device cuda:7 &
# done

# wait

# n=2000
# nodes=(10 40 60 80 100 200 400)
# for d in "${nodes[@]}"; do
#     for i in {1..5}; do
#         s0=$(( d * 4 ))
#         python gen_copies.py \
#         --gen_type W_torch \
#         --n $n --s0 $s0 --d $d \
#         --seed_knockoff $i \
#         --root_path simulated_data/v11 \
#         --version $d \
#         --device cuda:6 &
#     done

#     for i in {6..10}; do
#         s0=$(( d * 4 ))
#         python gen_copies.py \
#         --gen_type W_torch \
#         --n $n --s0 $s0 --d $d \
#         --seed_knockoff $i \
#         --root_path simulated_data/v11 \
#         --version $d \
#         --device cuda:7 &
#     done
#     wait
# done


############################################
# FDR control model fitting with dag control
############################################

# log_file_column=25
# log_file_global=$((log_file_column + 1))

# items=(1 3 5 7)
# for i in "${items[@]}"; do

#     python multi_main.py \
#     --d=$d \
#     --control_type=type_4 \
#     --seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
#     --seed_model_list=0 \
#     --dag_control=dag_${i} \
#     --version=$version \
#     --log_file=log_${log_file_column}_${i} &

#     python multi_main.py \
#     --d=$d \
#     --control_type=type_4_global \
#     --seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
#     --seed_model_list=0 \
#     --dag_control=dag_${i} \
#     --version=$version \
#     --log_file=log_${log_file_global}_${i} &

# done

log_file_column=31
log_file_global=$((log_file_column + 1))

n=2000
# nodes=(10 40 60 80 100 200 400)
nodes=(10 40 60 80 100)
for d in "${nodes[@]}"; do
    s0=$(( d * 4 ))
    python multi_main.py \
    --n $n --s0 $s0 --d $d \
    --control_type=type_4 \
    --dag_control=dag_1 \
    --seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
    --seed_model_list=0 \
    --version=$d \
    --root_path simulated_data/v11 \
    --log_file=log_${log_file_column}/log_${d} &

    python multi_main.py \
    --n $n --s0 $s0 --d $d \
    --control_type=type_4_global \
    --dag_control=dag_1 \
    --seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
    --seed_model_list=0 \
    --version=$d \
    --root_path simulated_data/v11 \
    --log_file=log_${log_file_global}/log_${d} &
    wait
done

################################################
# FDR control model fitting without dag control
################################################

# log_file_column=27
# log_file_global=$((log_file_column + 1))

# python multi_main.py \
# --n $n --s0 $s0 --d $d \
# --control_type=type_3 \
# --seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
# --seed_model_list=0 \
# --version=$version \
# --log_file=log_${log_file_column} &

# python multi_main.py \
# --n $n --s0 $s0 --d $d \
# --control_type=type_3_global \
# --seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
# --seed_model_list=0 \
# --version=$version \
# --log_file=log_${log_file_global} &

# log_file_column=29
# log_file_global=$((log_file_column + 1))

# n=2000
# # nodes=(10 40 60 80 100 200 400)
# nodes=(10 40 60 80 100)
# for d in "${nodes[@]}"; do
#     s0=$(( d * 4 ))
#     python multi_main.py \
#     --n $n --s0 $s0 --d $d \
#     --control_type=type_3 \
#     --seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
#     --seed_model_list=0 \
#     --version=$d \
#     --root_path simulated_data/v11 \
#     --log_file=log_${log_file_column}/log_${d} &

#     python multi_main.py \
#     --n $n --s0 $s0 --d $d \
#     --control_type=type_3_global \
#     --seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
#     --seed_model_list=0 \
#     --version=$d \
#     --root_path simulated_data/v11 \
#     --log_file=log_${log_file_global}/log_${d} &
#     wait
# done