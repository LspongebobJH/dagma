# version=5
# log_file_global=14
# log_file_column=$((log_file_global + 1))

# python gen_copies.py --gen_type X --version $version
# for i in {1..10}; do
#     python gen_copies.py --gen_type knockoff --seed_knockoff $i --version $version &
# done

# wait

# for i in {1..10}; do
#     python gen_copies.py --gen_type W  --seed_knockoff $i --version $version &
# done

# wait

# if [ $? -eq 0 ]; then
    # python multi_main.py --control_type=type_4_global --dagma_type=dagma_1 \
    # --seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
    # --seed_model_list=0 \
    # --version=$version \
    # --log_file=log_15 &

    # python multi_main.py --control_type=type_3 --dagma_type=dagma_1 \
    # --seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
    # --seed_model_list=0 \
    # --version=$version \
    # --log_file=log_${log_file_column} &
# fi

# for i in {5..6}; do

#     python multi_main.py \
#     --s0 80 \
#     --control_type=type_4_global \
#     --seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
#     --seed_model_list=0 \
#     --dag_control=dag_${i} \
#     --version=2 \
#     --log_file=log_16_${i} &

# done

python multi_main.py \
--s0 120 \
--control_type=type_4_global \
--seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
--seed_model_list=0 \
--dag_control=dag_5 \
--version=2 \
--log_file=log_15_7 &

python multi_main.py \
--s0 120 \
--control_type=type_4_global \
--seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
--seed_model_list=0 \
--dag_control=dag_6 \
--version=2 \
--log_file=log_15_8 &

python multi_main.py \
--s0 80 \
--control_type=type_4_global \
--seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
--seed_model_list=0 \
--dag_control=dag_5 \
--version=5 \
--log_file=log_16_7 &

python multi_main.py \
--s0 80 \
--control_type=type_4_global \
--seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
--seed_model_list=0 \
--dag_control=dag_6 \
--version=5 \
--log_file=log_16_8 &