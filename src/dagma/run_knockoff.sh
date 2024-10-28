#!/bin/bash

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

echo "Start generating knockoff from 1 to 3..." > /home/jiahang/dagma/src/dagma/pipe_log.log

X_version=11

# dst_data_version=47 # dagma
# dst_data_version=49 # tree
# dst_data_version=51 # notears
# dst_data_version=53 # golem(-EV)
dst_data_version=55 # dag_gnn

# fit_W_version=39 # dagma
# fit_W_version=48 # tree
# fit_W_version=50 # notears
# fit_W_version=52 # golem(-EV)
fit_W_version=54 # dag_gnn

n=2000
cuda_idx=5
nodes=(20 40 60)
s0_factors=(4 6)
seedsX=( {1..10..1} )
seedsKnockoff=( 1 )
suffixs=(_normX=sym1_option=5_knock=OLS
         _normX=sym1_option=10_topo_knock=PLS
        )
cnt=0

for d in "${nodes[@]}"; do
    for s0_factor in "${s0_factors[@]}"; do
        for suffix in "${suffixs[@]}"; do
            s0=$(( d * s0_factor ))
            
            dst_data_name=${n}_${d}_${s0}${suffix}
            X_name=${n}_${d}_${s0}_normX_sym1
            
            ./create_data_dir.sh X $dst_data_version $dst_data_name $X_version $X_name

            target_dir=/home/jiahang/dagma/src/dagma/simulated_data/v${dst_data_version}/v$dst_data_name/knockoff
            if [ ! -d "$target_dir$" ]; then
                mkdir -p ${target_dir}
            fi

            for seedKnockoff in "${seedsKnockoff[@]}"; do
                for seedX in "${seedsX[@]}"; do

                    fit_W_name=W_${seedX}_0_normX=sym1

                    if [ $suffix = '_normX=sym1_option=5_knock=OLS' ]; then
                        CUDA_VISIBLE_DEVICES=${cuda_idx} \
                        python knockoff_v44.py \
                        --W_type=W_est \
                        --dst_data_version=v${dst_data_version} \
                        --dst_data_name=v${dst_data_name} \
                        --fit_W_version=v${fit_W_version} \
                        --fit_W_name=${fit_W_name} \
                        --method_diagn_gen=OLS_cuda \
                        --option=5 \
                        --force_save \
                        --n=${n} --d=${d} --s0=${s0} \
                        --dedup \
                        --device=cuda:${cuda_idx} \
                        --seed_X=${seedX} \
                        --seed_knockoff=${seedKnockoff} \
                        >/home/jiahang/dagma/src/dagma/simulated_data/v${dst_data_version}/v$dst_data_name/knockoff/log_${seedX}_${seedKnockoff} 2>&1 &
                    elif [ $suffix = '_normX=sym1_option=10_topo_knock=PLS' ]; then
                        CUDA_VISIBLE_DEVICES=${cuda_idx} \
                        python knockoff_v44.py \
                        --W_type=W_est \
                        --dst_data_version=v${dst_data_version} \
                        --dst_data_name=v${dst_data_name} \
                        --fit_W_version=v${fit_W_version} \
                        --fit_W_name=${fit_W_name} \
                        --method_diagn_gen=PLS \
                        --option=10 \
                        --topo_sort \
                        --force_save \
                        --n=${n} --d=${d} --s0=${s0} \
                        --dedup \
                        --device=cuda:${cuda_idx} \
                        --seed_X=${seedX} \
                        --seed_knockoff=${seedKnockoff} \
                        >/home/jiahang/dagma/src/dagma/simulated_data/v${dst_data_version}/v$dst_data_name/knockoff/log_${seedX}_${seedKnockoff} 2>&1 &
                    fi

                        # --note="_grnboost2" \

                    # control cuda device
                    cuda_idx=$(( cuda_idx + 1))
                    if [ $cuda_idx -eq 8 ]; then
                        cuda_idx=5
                    fi
                    
                    # control parallel process number
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

wait

echo "End generating knockoff from 1 to 3." > /home/jiahang/dagma/src/dagma/pipe_log.log