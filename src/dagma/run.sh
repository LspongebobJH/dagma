version=8
s0=200
d=50
n=1000

# python gen_copies.py --gen_type X --n $n --s0 $s0 --d $d --version $version
# for i in {1..10}; do
#     python gen_copies.py --gen_type knockoff --n $n --s0 $s0 --d $d  --seed_knockoff $i --version $version &
# done

# wait

for i in {1..10}; do
    python gen_copies.py --gen_type W --n $n --s0 $s0 --d $d  --seed_knockoff $i --version $version
done

wait

# log_file_column=23
# log_file_global=$((log_file_column + 1))

# items=(1 3 5 7)
# for i in "${items[@]}"; do

#     python multi_main.py \
#     --s0 $s0 \
#     --control_type=type_3 \
#     --seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
#     --seed_model_list=0 \
#     --dag_control=dag_${i} \
#     --version=$version \
#     --log_file=log_${log_file_column}_${i} &

#     python multi_main.py \
#     --s0 $s0 \
#     --control_type=type_3_global \
#     --seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
#     --seed_model_list=0 \
#     --dag_control=dag_${i} \
#     --version=$version \
#     --log_file=log_${log_file_global}_${i} &

# done