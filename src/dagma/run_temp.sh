#!/bin/bash
data_version_list=(20_1 21_1)
for _version in "${data_version_list[@]}"; do
    # ./create_data_dir.sh ${_version} 40 11 X && ./create_data_dir.sh ${_version} 40 11 knockoff
    # rm -r /home/jiahang/dagma/src/dagma/simulated_data/v${_version}
done