# python gen_copies.py --gen_type X
# for i in {1..10}; do
# 	python gen_copies.py --gen_type W  --seed_knockoff $i &
# done

python multi_main.py --control_type=type_3_global --dagma_type=dagma_1 \
--seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
--seed_model_list=0 \
--version=1 \
--log_file log_3 &

python multi_main.py --control_type=type_3 --dagma_type=dagma_1 \
--seed_knockoff_list=1,2,3,4,5,6,7,8,9,10 \
--seed_model_list=0 \
--version=1 \
--log_file log_4 &