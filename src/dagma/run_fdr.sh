#!/bin/bash

############################################
# FDR control selection with dag control
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

################################################################
# FDR control selection with dag control, sweep over n_nodes
################################################################

# log_file_column=31
# log_file_global=$((log_file_column + 1))

# n=2000
# # nodes=(10 40 60 80 100 200 400)
# nodes=(10 40 60 80 100)
# for d in "${nodes[@]}"; do
#     s0=$(( d * 4 ))
#     python multi_main.py \
#     --n $n --s0 $s0 --d $d \
#     --control_type=type_4 \
#     --dag_control=dag_1 \
#     --seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
#     --seed_model_list=0 \
#     --version=$d \
#     --root_path simulated_data/v11 \
#     --log_file=log_${log_file_column}/log_${d} &

#     python multi_main.py \
#     --n $n --s0 $s0 --d $d \
#     --control_type=type_4_global \
#     --dag_control=dag_1 \
#     --seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
#     --seed_model_list=0 \
#     --version=$d \
#     --root_path simulated_data/v11 \
#     --log_file=log_${log_file_global}/log_${d} &
#     wait
# done

################################################
# FDR control selection without dag control
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

####################################################################
# FDR control selection without dag control, sweep over n_nodes
####################################################################

run() {
    # data_version=44
    data_version=47
    log_file_global=99
    # options=( 7 9 1 2 10 11 )
    # options=(1 2 7 9 10 11 12 )
    options=(5)
    # option=5
    n=2000
    
    nodes=(20 40 60 80 100)
    for d in "${nodes[@]}"; do
        for option in "${options[@]}"; do
            s0=$(( d * 6 ))

            # version=${d}_${s0}_condX_5e4_option_5_OLS
            # python multi_main.py \
            # --n $n --s0 $s0 --d $d \
            # --control_type=type_3_global \
            # --seed_X_list=3,4,5,6,11,12,13,15,17,18 \
            # --seed_knockoff_list=1 \
            # --seed_model_list=0 \
            # --version=$version \
            # --root_path simulated_data/v${data_version} \
            # --n_jobs=4 \
            # --log_file=log_${log_file_global}/log_${data_version}_${version} &
            # wait

            version=${d}_${s0}_normX_sym1_option_${option}_OLS_new
            python multi_main.py \
            --n $n --s0 $s0 --d $d \
            --control_type=type_3_global \
            --seed_X_list=1,2,3,4,5,6,7,8,9,10 \
            --seed_knockoff_list=1 \
            --seed_model_list=0 \
            --version=$version \
            --root_path simulated_data/v${data_version} \
            --n_jobs=4 \
            --log_file=log_${log_file_global}/log_${data_version}_${version}_1-10 &

            # version=${d}_${s0}_normX_B1Col_option_5_OLS
            # python multi_main.py \
            # --n $n --s0 $s0 --d $d \
            # --control_type=type_3_global \
            # --seed_X_list=1,2,7,8,9,10,14,16,19,23 \
            # --seed_knockoff_list=1 \
            # --seed_model_list=0 \
            # --version=$version \
            # --root_path simulated_data/v${data_version} \
            # --n_jobs=4 \
            # --log_file=log_${log_file_global}/log_${data_version}_${version} &
            # # wait
        done
    done

}

run