#!/bin/bash

data_version=$1
n_nodes=$2
src_data_version=$3
src_data_option=$4

dst_dir=/home/jiahang/dagma/src/dagma/simulated_data/v${data_version}/v${n_nodes}
src_path=/home/jiahang/dagma/src/dagma/simulated_data/v${src_data_version}/v${n_nodes}/${src_data_option}

mkdir -p ${dst_dir}
ln -snf ${src_path} ${dst_dir}/${src_data_option}