version=7
s0=1000
d=60
n=2000

########################################
# Generating X
########################################

# python gen_copies.py --gen_type X --n $n --s0 $s0 --d $d --version $version

########################################
# Generating Knockoff
########################################

# for i in {1..10}; do
#     python gen_copies.py --gen_type knockoff --n $n --s0 $s0 --d $d  --seed_knockoff $i --version $version &
# done

# wait

########################################
# Generating W
########################################

# for i in {1..5}; do
#     python gen_copies.py \
#     --gen_type W_torch \
#     --n $n --s0 $s0 --d $d \
#     --seed_knockoff $i --version $version \
#     --device cuda:6 &
# done

# for i in {6..10}; do
#     python gen_copies.py \
#     --gen_type W_torch \
#     --n $n --s0 $s0 --d $d \
#     --seed_knockoff $i --version $version \
#     --device cuda:7 &
# done

# wait

########################################
# FDR control model fitting
########################################

log_file_column=25
log_file_global=$((log_file_column + 1))

items=(1 3 5 7)
for i in "${items[@]}"; do

    python multi_main.py \
    --d=$d \
    --control_type=type_4 \
    --seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
    --seed_model_list=0 \
    --dag_control=dag_${i} \
    --version=$version \
    --log_file=log_${log_file_column}_${i} &

    python multi_main.py \
    --d=$d \
    --control_type=type_4_global \
    --seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
    --seed_model_list=0 \
    --dag_control=dag_${i} \
    --version=$version \
    --log_file=log_${log_file_global}_${i} &

done