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

# n=2000
# d=20
# s0=40

cnt=0
ns=(2000)
nodes=(20)
s0_factors=(4 6)
seedsX=( {1..10..1} )
for n in "${ns[@]}"; do
    for d in "${nodes[@]}"; do
        for s0_factor in "${s0_factors[@]}"; do
            for seedX in "${seedsX[@]}"; do
                s0=$(( d * s0_factor ))
                
                python gen_copies.py --gen_type X \
                --n $n --d $d --s0 $s0 \
                --graph_type ER \
                --norm_data_gen sym_1 \
                --force_save \
                --seed_X $seedX \
                --root_path simulated_data/v11 \
                --version ${n}_${d}_${s0}_normX_sym1 &

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

echo "End generating X from 1 to 10." >> /home/jiahang/dagma/src/dagma/pipe_log.log

# # wait

#########################################
# Misc
#########################################

# python test.py --exp_group_idx=v43 --d=20 --v43_method='elastic' --v43_disable_dag_control --device=cuda:6 &
# python test.py --exp_group_idx=v43 --d=100 --v43_method='elastic' --v43_disable_dag_control --device=cuda:7 &
# python test.py --exp_group_idx=v44 --v44_option=1 --lasso_alpha=OLS --W_type=W_true \
#     --d=20 --device=cuda:5 &

