#!/bin/bash

########################################
# Generating W (W_torch)
########################################

# n=2000
# d=20
# s0=80
# version=11
# root_path=simulated_data/v$version

# for i in {1..5}; do
#     python gen_copies.py \
#     --gen_type W_torch \
#     --n $n --s0 $s0 --d $d \
#     --seed_knockoff $i --version $d \
#     --root_path $root_path \
#     --device cuda:6 &
# done

# for i in {6..10}; do
#     python gen_copies.py \
#     --gen_type W_torch \
#     --n $n --s0 $s0 --d $d \
#     --seed_knockoff $i --version $d \
#     --root_path $root_path \
#     --device cuda:7 &
# done

# wait

############################################
# Generating W (W_torch), sweep over n_nodes
############################################

run() {
    dst_data_version=$1
    d=$2
    cuda_idx=$3

    # src_data_version=11

    n=2000
    s0=$(( d * 6 ))
    suffixs=(_normX_sym1_option_10_PLS_topo_sort _normX_sym1_option_10_PLS_topo_sort_nComp_3)
    seedsX=( {1..10..1} )
    seedsKnockoff=( 1 )
    # options=( 5 )
    # nComps=( 3 4 )
    cnt=0


    # for version in "${version_list[@]}"; do
    for suffix in "${suffixs[@]}"; do
        # for nComp in "${nComps[@]}"; do
            # version=${d}_${s0}_normX_sym1_option_${option}_PLS
            version=${d}_${s0}${suffix}
            # ./create_data_dir.sh X $dst_data_version $version $src_data_version ${d}

            target_dir=/home/jiahang/dagma/src/dagma/simulated_data/v${dst_data_version}/v$version/W
            if [ ! -d "$target_dir$" ]; then
                mkdir -p ${target_dir}
            fi

            # for (( seedX=1; seedX<=100; seedX++ )); do
            for seedKnockoff in "${seedsKnockoff[@]}"; do
                for seedX in "${seedsX[@]}"; do
                    stdbuf -o0 -e0 \
                    python gen_copies.py \
                    --gen_type W_genie3 \
                    --n $n --s0 $s0 --d $d \
                    --seed_X $seedX \
                    --seed_knockoff $seedKnockoff \
                    --root_path simulated_data/v${dst_data_version} \
                    --version ${version} \
                    --force_save \
                    --device cuda:${cuda_idx} > simulated_data/v${dst_data_version}/v${version}/W/log_${seedX}_${seedKnockoff}_0 2>&1 &

                    cuda_idx=$(( cuda_idx + 1))
                    if [ $cuda_idx -eq 8 ]; then
                        cuda_idx=5
                    fi

                    cnt=$(( cnt + 1 ))
                    _cnt=$(( cnt % 20 ))
                    
                    if [ ${_cnt} -eq 19 ]; then
                        wait
                    fi

                    # _seedX=$(( seedX % 20 ))
                    # if [ ${_seedX} -eq 10 ]; then
                    #     wait
                    # fi
                done
            done
        # done
    done
    
}

data_version=49


# d=20
# cuda_idx=5
# run $data_version $d $cuda_idx

# # wait

# d=40
# cuda_idx=6
# run $data_version $d $cuda_idx

# # wait

# d=60
# cuda_idx=7
# run $data_version $d $cuda_idx

# wait

# d=80
# cuda_idx=5
# run $data_version $d $cuda_idx


d=100
cuda_idx=7
run $data_version $d $cuda_idx

