#!/bin/bash

data_option=$1

dst_data_version=$2
dst_final_version=$3

src_data_version=$4
src_final_version=$5

dst_dir=/home/jiahang/dagma/src/dagma/simulated_data/v${dst_data_version}/v${dst_final_version}
dst_path=$dst_dir/$data_option
src_path=/home/jiahang/dagma/src/dagma/simulated_data/v${src_data_version}/v${src_final_version}/${data_option}


if [ ! -d "$dst_dir$" ]; then
    mkdir -p ${dst_dir}
fi

if [ ! -f "$dst_path/$" ]; then
    ln -snf ${src_path} ${dst_path}
fi

