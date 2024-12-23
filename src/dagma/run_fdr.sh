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
    data_version=55
    log_file_global=103
    suffixs=(
        _normX=sym1_option=5_knock=PLS
    )
    control=global
    ns=(2000)
    nodes=(40)
    s0_factors=(6)
    lambda_l1s=(0.1 0.5 0.9)
    lambda_l2s=(0.1 0.5 0.9)
    for lambda_l1 in "${lambda_l1s[@]}"; do
        for lambda_l2 in "${lambda_l2s[@]}"; do
            for n in "${ns[@]}"; do
                for d in "${nodes[@]}"; do
                    for s0_factor in "${s0_factors[@]}"; do

                        for suffix in "${suffixs[@]}"; do

                            s0=$(( d * s0_factor ))

                            # version=${d}_${s0}${suffix}
                            # version=${n}_${d}_${s0}${suffix}
                            version=${n}_${d}_${s0}${suffix}_l1=${lambda_l1}_l2=${lambda_l2}
                            
                            python multi_main.py \
                            --n $n --s0 $s0 --d $d \
                            --control_type=type_3_global \
                            --seed_X_list=1 \
                            --seed_knockoff_list=1 \
                            --seed_model_list=0 \
                            --version=$version \
                            --root_path simulated_data/v${data_version} \
                            --n_jobs=4 \
                            --log_file=log_${log_file_global}/log_${version} &
                        done
                    done
                done
            done
        done
    done
    

}

echo "Start fdr control from 1 to 3..." >> /home/jiahang/dagma/src/dagma/pipe_log.log
run
wait
echo "End fdr control of [X, X'] from 1 to 3..." >> /home/jiahang/dagma/src/dagma/pipe_log.log