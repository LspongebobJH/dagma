#!/bin/bash

data_version=$1
n_nodes=$2
alpha=$3
src_data_version=$4
src_data_option=$5

# dst_dir=/home/jiahang/dagma/src/dagma/simulated_data/v${data_version}/v${n_nodes}
dst_dir=/home/jiahang/dagma/src/dagma/simulated_data/v${data_version}/v${data_version}_${alpha}/v${n_nodes}
dst_path=$dst_dir/$src_data_option
src_path=/home/jiahang/dagma/src/dagma/simulated_data/v${src_data_version}/v${n_nodes}/${src_data_option}


if [ ! -d "$dst_dir$" ]; then
    mkdir -p ${dst_dir}
fi

if [ ! -f "$dst_path/$" ]; then
    ln -snf ${src_path} ${dst_path}
fi

